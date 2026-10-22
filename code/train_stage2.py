"""
Stage 2 Training: DA-SAM
===========================
Train the full DA-SAM model with dual adapters
"""
import os
import csv
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from segment_anything import sam_model_registry

from config import opt
from models import UNetRefiner, AIA, WFA, DA_SAM
from datasets import MetalArtifactDataset
from utils import dice_loss, compute_metrics


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(cfg, device):
    """Build DA-SAM model"""
    stage1_ckpt_path = os.path.join(cfg.OUTPUT_DIR, "stage1_unet", "best.pt")
    if not os.path.isfile(stage1_ckpt_path):
        raise FileNotFoundError(
            f"Required Stage 1 checkpoint not found: {stage1_ckpt_path}. "
            "Train the mask prompt generator before starting Stage 2."
        )

    print(f"\n{'='*60}")
    print(f"Building DA-SAM with WFA (Wavelet Frequency Adapter)")
    print(f"Adapter dim: {cfg.ADAPTER_DIM}, Scale: {cfg.ADAPTER_SCALE}")
    print(f"{'='*60}\n")

    # Load SAM
    sam = sam_model_registry[cfg.MODEL_TYPE](checkpoint=cfg.SAM_CKPT)

    # Freeze SAM encoder and prompt encoder
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    # U-Net Refiner (trained Stage 1 weights are required)
    unet = UNetRefiner(in_ch=2, base_ch=cfg.UNET_BASE_CH, depth=cfg.UNET_DEPTH)
    ckpt = torch.load(stage1_ckpt_path, map_location=device)
    unet.load_state_dict(ckpt["unet"])
    print(f"Loaded U-Net from {stage1_ckpt_path}")

    # Freeze U-Net
    for p in unet.parameters():
        p.requires_grad = False

    # Image Adapter
    aia = AIA(in_ch=2, base_ch=cfg.AIA_CH, mod_scale=cfg.MOD_SCALE)

    # WFA adapters
    num_blocks = len(sam.image_encoder.blocks)
    embed_dim = sam.image_encoder.patch_embed.proj.out_channels
    freq_adapters = nn.ModuleList([
        WFA(embed_dim, cfg.ADAPTER_DIM, cfg.ADAPTER_SCALE) for _ in range(num_blocks)
    ])

    # Full model
    model = DA_SAM(sam, unet, aia, freq_adapters, cfg).to(device)

    # Optimizer with different learning rates
    params = [
        {"params": model.aia.parameters(), "lr": cfg.LR_AIA},
        {"params": model.freq_adapters.parameters(), "lr": cfg.LR_FREQ_ADAPTER},
        {"params": model.sam.mask_decoder.parameters(), "lr": cfg.LR_DECODER},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=cfg.WEIGHT_DECAY_STAGE2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.STAGE2_EPOCHS, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP)

    # Stats
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    freq_params = sum(p.numel() for p in model.freq_adapters.parameters())
    aia_params = sum(p.numel() for p in model.aia.parameters())

    print(f"[PARAMS] Total: {total / 1e6:.1f}M")
    print(f"[PARAMS] Trainable: {trainable / 1e6:.2f}M ({100 * trainable / total:.1f}%)")
    print(f"[PARAMS] WFA Adapters: {freq_params / 1e6:.3f}M ({100 * freq_params / trainable:.1f}% of trainable)")
    print(f"[PARAMS] AIA: {aia_params / 1e6:.3f}M")
    print()

    return model, optimizer, scheduler, scaler


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    """Evaluation function"""
    model.eval()
    total_loss = 0.0
    total_n = 0
    metrics_sum = {"iou": 0.0, "dice": 0.0, "prec": 0.0, "rec": 0.0}

    for batch in loader:
        img_ori = batch["img_ori"].to(device)
        img_eq = batch["img_eq"].to(device)
        mask_coarse = batch["mask_coarse"].to(device)
        gt = batch["gt"].to(device)

        pred, unet_logits, delta = model(img_ori, img_eq, mask_coarse)

        # Loss
        bce = F.binary_cross_entropy_with_logits(pred, gt)
        dl = dice_loss(pred, gt)
        aux = F.mse_loss(delta, gt * cfg.TARGET_MOD)
        loss = cfg.LAMBDA_BCE * bce + cfg.LAMBDA_DICE * dl + cfg.LAMBDA_AUX * aux

        # Metrics
        metrics = compute_metrics(pred, gt)

        bs = gt.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        for k in metrics_sum:
            metrics_sum[k] += metrics[k] * bs

    model.train()
    return total_loss / total_n, {k: v / total_n for k, v in metrics_sum.items()}


def train(cfg):
    """Main training function for Stage 2"""
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    stage2_dir = os.path.join(cfg.OUTPUT_DIR, "stage2_wfa")
    os.makedirs(stage2_dir, exist_ok=True)

    # Load datasets
    train_ds = MetalArtifactDataset(cfg.TRAIN_NAMES, cfg, "train")
    val_ds = MetalArtifactDataset(cfg.VAL_NAMES, cfg, "val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.STAGE2_BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.STAGE2_BATCH_SIZE, num_workers=cfg.NUM_WORKERS
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Build model
    model, optimizer, scheduler, scaler = build_model(cfg, device)

    # Logging
    writer = SummaryWriter(os.path.join(stage2_dir, "tb"))
    csv_file = open(os.path.join(stage2_dir, "log.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "dice", "iou", "prec", "rec", "time"])

    # Save config
    config_path = os.path.join(stage2_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write("DA-SAM with WFA (Wavelet Frequency Adapter)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Adapter Dim: {cfg.ADAPTER_DIM}\n")
        f.write(f"Adapter Scale: {cfg.ADAPTER_SCALE}\n")
        f.write(f"AIA CH: {cfg.AIA_CH}\n")
        f.write(f"Batch Size: {cfg.STAGE2_BATCH_SIZE}\n")
        f.write(f"Epochs: {cfg.STAGE2_EPOCHS}\n")

    # Training loop
    best_dice = 0.0
    patience_cnt = cfg.STAGE2_PATIENCE
    step = 0

    print(f"\n{'='*60}")
    print(f"Stage 2: DA-SAM Training")
    print(f"{'='*60}")
    print(f"Output: {stage2_dir}")
    print(f"Epochs: {cfg.STAGE2_EPOCHS}, Batch size: {cfg.STAGE2_BATCH_SIZE}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.STAGE2_EPOCHS + 1):
        t0 = time.time()
        total_loss = 0.0
        total_n = 0

        model.train()

        for it, batch in enumerate(train_loader):
            img_ori = batch["img_ori"].to(device)
            img_eq = batch["img_eq"].to(device)
            mask_coarse = batch["mask_coarse"].to(device)
            gt = batch["gt"].to(device)

            # Forward
            with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
                pred, unet_logits, delta = model(img_ori, img_eq, mask_coarse)
                bce = F.binary_cross_entropy_with_logits(pred, gt)
                dl = dice_loss(pred, gt)
                aux = F.mse_loss(delta, gt * cfg.TARGET_MOD)
                loss = cfg.LAMBDA_BCE * bce + cfg.LAMBDA_DICE * dl + cfg.LAMBDA_AUX * aux

            # Backward
            scaler.scale(loss / cfg.ACCUM_STEPS).backward()
            total_loss += loss.item() * gt.size(0)
            total_n += gt.size(0)

            if (it + 1) % cfg.ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.CLIP_NORM
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.LOG_EVERY == 0:
                    writer.add_scalar("train/loss", loss.item(), step)
                if step % cfg.PRINT_EVERY == 0:
                    print(f"  [Ep {epoch:02d} Step {step:05d}] loss={loss.item():.4f}")

        # Validation
        val_loss, val_m = evaluate(model, val_loader, device, cfg)
        scheduler.step()
        train_loss = total_loss / total_n
        elapsed = time.time() - t0

        # Logging
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_dice", val_m['dice'], epoch)
        writer.add_scalar("epoch/val_iou", val_m['iou'], epoch)

        csv_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{val_m['dice']:.6f}", f"{val_m['iou']:.6f}",
            f"{val_m['prec']:.6f}", f"{val_m['rec']:.6f}", f"{elapsed:.1f}"
        ])
        csv_file.flush()

        print(f"[Epoch {epoch:02d}] train={train_loss:.4f} val={val_loss:.4f} "
              f"dice={val_m['dice']:.4f} iou={val_m['iou']:.4f} time={elapsed:.1f}s")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "adapter_type": "WFA",
            "adapter_dim": cfg.ADAPTER_DIM,
            "aia": model.aia.state_dict(),
            "freq_adapters": model.freq_adapters.state_dict(),
            "decoder": model.sam.mask_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(ckpt, os.path.join(stage2_dir, "last.pt"))

        # Best model
        if val_m['dice'] > best_dice + 1e-6:
            best_dice = val_m['dice']
            patience_cnt = cfg.STAGE2_PATIENCE
            ckpt["best_dice"] = best_dice
            torch.save(ckpt, os.path.join(stage2_dir, "best.pt"))
            print(f"  >> New best dice: {best_dice:.4f}")
        else:
            patience_cnt -= 1
            if patience_cnt <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    # Cleanup
    csv_file.close()
    writer.close()

    print(f"\nStage 2 complete!")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {stage2_dir}")

    return best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Train DA-SAM")
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--root', type=str, default=None, help='Project root')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')

    args = parser.parse_args()

    # Update config
    if args.root:
        opt.update_paths(root=args.root)
    if args.data_dir:
        opt.update_paths(data_dir=args.data_dir)
    if args.batch_size:
        opt.STAGE2_BATCH_SIZE = args.batch_size
    if args.epochs:
        opt.STAGE2_EPOCHS = args.epochs

    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)

    train(opt)
