"""
Stage 2 Testing: DA-SAM
"""
import os
import csv
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from config import opt
from models import UNetRefiner, AIA, WFA, DA_SAM
from datasets import InferenceDataset
from utils import get_full_metrics


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg, device):
    """Build DA-SAM model for testing/inference"""
    sam = sam_model_registry[cfg.MODEL_TYPE](checkpoint=cfg.SAM_CKPT)

    unet = UNetRefiner(in_ch=2, base_ch=cfg.UNET_BASE_CH, depth=cfg.UNET_DEPTH)
    stage1_ckpt = os.path.join(cfg.OUTPUT_DIR, "stage1_unet", "best.pt")
    if os.path.exists(stage1_ckpt):
        ckpt = torch.load(stage1_ckpt, map_location=device)
        unet.load_state_dict(ckpt["unet"])
        print(f"Loaded U-Net from {stage1_ckpt}")

    aia = AIA(in_ch=2, base_ch=cfg.AIA_CH, mod_scale=cfg.MOD_SCALE)

    num_blocks = len(sam.image_encoder.blocks)
    embed_dim = sam.image_encoder.patch_embed.proj.out_channels
    freq_adapters = nn.ModuleList([
        WFA(embed_dim, cfg.ADAPTER_DIM, cfg.ADAPTER_SCALE) for _ in range(num_blocks)
    ])

    model = DA_SAM(sam, unet, aia, freq_adapters, cfg).to(device)

    stage2_ckpt = os.path.join(cfg.OUTPUT_DIR, "stage2_wfa", "best.pt")
    if os.path.exists(stage2_ckpt):
        ckpt = torch.load(stage2_ckpt, map_location=device)
        model.aia.load_state_dict(ckpt["aia"])
        model.freq_adapters.load_state_dict(ckpt["freq_adapters"])
        model.sam.mask_decoder.load_state_dict(ckpt["decoder"])
        print(f"Loaded DA-SAM from {stage2_ckpt}")
        if "best_dice" in ckpt:
            print(f"Checkpoint best dice: {ckpt['best_dice']:.4f}")

    model.eval()
    return model


@torch.no_grad()
def evaluate(cfg, save_predictions=False):
    """Evaluate DA-SAM on the test split and optionally save predictions."""
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_ds = InferenceDataset(cfg.TEST_NAMES, cfg)
    print(f"Test samples: {len(test_ds)}")

    model = build_model(cfg, device)

    pred_dir = None
    csv_path = None
    if save_predictions:
        eval_dir = os.path.join(cfg.OUTPUT_DIR, "eval_results")
        os.makedirs(eval_dir, exist_ok=True)
        pred_dir = os.path.join(eval_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        csv_path = os.path.join(eval_dir, "DA_SAM_Test_Metrics.csv")

    rows = []
    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    hd95_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    n = 0

    print("=" * 90)
    print(f"{'File Name':<30} | {'Dice':<8} | {'IoU':<8} | {'HD95':<8} | {'Prec':<8} | {'Rec':<8}")
    print("-" * 90)

    for idx in range(len(test_ds)):
        batch = test_ds[idx]
        sample = batch["sample"]

        img_ori = batch["img_ori"].to(device)
        img_eq = batch["img_eq"].to(device)
        mask = batch["mask"].to(device)
        gt = torch.from_numpy(batch["gt"]).float().unsqueeze(0).unsqueeze(0).to(device)

        pred, _, _ = model(img_ori, img_eq, mask)

        bce = F.binary_cross_entropy_with_logits(pred, gt)
        dl = 1.0 - (2 * (torch.sigmoid(pred) * gt).sum(dim=(2, 3)) + 1e-6) / (
            torch.sigmoid(pred).sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) + 1e-6
        )
        loss = cfg.LAMBDA_BCE * bce + cfg.LAMBDA_DICE * dl.mean()

        pred_prob = torch.sigmoid(pred).cpu().numpy()[0, 0]
        h, w = batch["orig_hw"]
        pred_binary = (pred_prob[:h, :w] > cfg.THRESHOLD).astype(np.float32)
        gt_cropped = batch["gt"][:h, :w]

        dice, iou, hd95, prec, rec = get_full_metrics(pred_binary, gt_cropped)
        fname = f"{sample['name']}_{sample['fname']}"
        rows.append([fname, dice, iou, hd95, prec, rec])

        loss_sum += loss.item()
        dice_sum += dice
        iou_sum += iou
        hd95_sum += hd95
        prec_sum += prec
        rec_sum += rec
        n += 1

        print(f"{fname:<30} | {dice:8.4f} | {iou:8.4f} | {hd95:8.2f} | {prec:8.4f} | {rec:8.4f}")

        if save_predictions:
            sample_dir = os.path.join(pred_dir, sample["name"])
            os.makedirs(sample_dir, exist_ok=True)
            save_path = os.path.join(sample_dir, sample["fname"])
            cv2.imwrite(save_path, (pred_binary * 255).astype(np.uint8))

    if n == 0:
        print("No valid test samples found.")
        return None

    avg = {
        "loss": loss_sum / n,
        "dice": dice_sum / n,
        "iou": iou_sum / n,
        "hd95": hd95_sum / n,
        "precision": prec_sum / n,
        "recall": rec_sum / n,
    }

    print("-" * 90)
    print(f"{'AVERAGE':<30} | {avg['dice']:8.4f} | {avg['iou']:8.4f} | {avg['hd95']:8.2f} | {avg['precision']:8.4f} | {avg['recall']:8.4f}")
    print("=" * 90)
    print(f"Loss: {avg['loss']:.4f}")

    if save_predictions and csv_path is not None:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['FileName', 'Dice', 'IoU', 'HD95', 'Precision', 'Recall'])
            writer.writerows(rows)
            writer.writerow(['AVERAGE', avg['dice'], avg['iou'], avg['hd95'], avg['precision'], avg['recall']])
        print(f"Predictions saved to: {pred_dir}")
        print(f"Metrics saved to: {csv_path}")

    return avg


def main(cfg=None):
    parser = argparse.ArgumentParser(description="Stage 2 Testing / Inference")
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--root', type=str, default=None, help='Project root')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--threshold', type=float, default=None, help='Prediction threshold')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction masks and CSV metrics')

    args = parser.parse_args()

    cfg = opt if cfg is None else cfg
    if args.root:
        cfg.update_paths(root=args.root)
    if args.data_dir:
        cfg.update_paths(data_dir=args.data_dir)
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.threshold is not None:
        cfg.THRESHOLD = args.threshold

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)

    evaluate(cfg, save_predictions=args.save_predictions)


if __name__ == "__main__":
    main()
