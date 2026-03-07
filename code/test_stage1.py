"""
Stage 1 Testing: U-Net Refiner
"""
import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import opt
from models import UNetRefiner
from datasets import Stage1Dataset
from utils import compute_metrics


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def test(cfg):
    """Test U-Net model"""
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    test_ds = Stage1Dataset(cfg.TEST_NAMES, cfg, "test")
    test_loader = DataLoader(test_ds, batch_size=cfg.STAGE1_BATCH_SIZE, num_workers=cfg.NUM_WORKERS)

    print(f"Test samples: {len(test_ds)}")

    # Load model
    model = UNetRefiner(in_ch=2, base_ch=cfg.UNET_BASE_CH, depth=cfg.UNET_DEPTH).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, "stage1_unet", "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["unet"])
    print(f"Loaded checkpoint from {ckpt_path}")

    if "best_dice" in ckpt:
        print(f"Checkpoint best dice: {ckpt['best_dice']:.4f}")

    # Test
    model.eval()
    dice_sum = 0.0
    iou_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in test_loader:
            img = batch["img"].to(device)
            mask = batch["mask"].to(device)
            gt = batch["gt"].to(device)

            pred = model(img, mask)
            metrics = compute_metrics(pred, gt)

            dice_sum += metrics["dice"] * gt.size(0)
            iou_sum += metrics["iou"] * gt.size(0)
            n += gt.size(0)

    avg_dice = dice_sum / n
    avg_iou = iou_sum / n

    print(f"\n{'='*40}")
    print(f"Test Results")
    print(f"{'='*40}")
    print(f"Dice: {avg_dice:.4f}")
    print(f"IoU: {avg_iou:.4f}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Testing")
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--root', type=str, default=None, help='Project root')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')

    args = parser.parse_args()

    if args.root:
        opt.update_paths(root=args.root)
    if args.data_dir:
        opt.update_paths(data_dir=args.data_dir)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)

    test(opt)
