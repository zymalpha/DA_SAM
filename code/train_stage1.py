"""
Stage 1 Training: U-Net Refiner
================================
Train U-Net for coarse mask refinement using BCE + Dice loss
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

from config import opt
from models import UNetRefiner
from datasets import Stage1Dataset
from utils import dice_loss, compute_metrics


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg):
    """Main training function for Stage 1"""
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    stage1_dir = os.path.join(cfg.OUTPUT_DIR, "stage1_unet")
    os.makedirs(stage1_dir, exist_ok=True)

    # Load datasets
    train_ds = Stage1Dataset(cfg.TRAIN_NAMES, cfg, "train")
    val_ds = Stage1Dataset(cfg.VAL_NAMES, cfg, "val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.STAGE1_BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.STAGE1_BATCH_SIZE, num_workers=cfg.NUM_WORKERS
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Create model
    model = UNetRefiner(in_ch=2, base_ch=cfg.UNET_BASE_CH, depth=cfg.UNET_DEPTH).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net parameters: {total_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.STAGE1_EPOCHS, eta_min=1e-6
    )

    # Loss
    train_criterion = nn.BCEWithLogitsLoss()

    # Logging
    writer = SummaryWriter(os.path.join(stage1_dir, "tb"))
    csv_file = open(os.path.join(stage1_dir, "log.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "time"])

    # Training loop
    best_dice = 0.0
    patience_cnt = cfg.STAGE1_PATIENCE

    print(f"\n{'='*60}")
    print(f"Stage 1: U-Net Refiner Training")
    print(f"{'='*60}")
    print(f"Output: {stage1_dir}")
    print(f"Epochs: {cfg.STAGE1_EPOCHS}, Batch size: {cfg.STAGE1_BATCH_SIZE}")
    print(f"Learning rate: {cfg.LR}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.STAGE1_EPOCHS + 1):
        t0 = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_n = 0

        for batch in train_loader:
            img = batch["img"].to(device)
            mask = batch["mask"].to(device)
            gt = batch["gt"].to(device)

            # Forward
            pred = model(img, mask)

            # Loss
            bce = train_criterion(pred, gt)
            dl = dice_loss(pred, gt)
            loss = cfg.LAMBDA_BCE * bce + cfg.LAMBDA_DICE * dl

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * gt.size(0)
            train_n += gt.size(0)

        train_loss = train_loss / train_n

        # Validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        dice_sum = 0.0
        iou_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                img = batch["img"].to(device)
                mask = batch["mask"].to(device)
                gt = batch["gt"].to(device)

                pred = model(img, mask)

                bce = train_criterion(pred, gt)
                dl = dice_loss(pred, gt)
                loss = cfg.LAMBDA_BCE * bce + cfg.LAMBDA_DICE * dl

                metrics = compute_metrics(pred, gt)

                val_loss += loss.item() * gt.size(0)
                val_n += gt.size(0)
                dice_sum += metrics["dice"] * gt.size(0)
                iou_sum += metrics["iou"] * gt.size(0)

        val_loss = val_loss / val_n
        val_dice = dice_sum / val_n
        val_iou = iou_sum / val_n

        scheduler.step()
        elapsed = time.time() - t0

        # Logging
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/dice", val_dice, epoch)
        writer.add_scalar("val/iou", val_iou, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        csv_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{val_dice:.6f}", f"{val_iou:.6f}", f"{elapsed:.1f}"
        ])
        csv_file.flush()

        print(f"[Epoch {epoch:03d}] train={train_loss:.4f} val={val_loss:.4f} "
              f"dice={val_dice:.4f} iou={val_iou:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
              f"time={elapsed:.1f}s")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "unet": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
            "config": cfg._state_dict()
        }
        torch.save(ckpt, os.path.join(stage1_dir, "last.pt"))

        # Best model
        if val_dice > best_dice + 1e-6:
            best_dice = val_dice
            patience_cnt = cfg.STAGE1_PATIENCE
            ckpt["best_dice"] = best_dice
            torch.save(ckpt, os.path.join(stage1_dir, "best.pt"))
            print(f"  >> New best: {best_dice:.4f}")
        else:
            patience_cnt -= 1
            if patience_cnt <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    # Cleanup
    csv_file.close()
    writer.close()

    print(f"\nStage 1 complete!")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {stage1_dir}")

    return best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Train U-Net Refiner")
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--root', type=str, default=None, help='Project root')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')

    args = parser.parse_args()

    # Update config
    if args.root:
        opt.update_paths(root=args.root)
    if args.data_dir:
        opt.update_paths(data_dir=args.data_dir)
    if args.batch_size:
        opt.STAGE1_BATCH_SIZE = args.batch_size
    if args.epochs:
        opt.STAGE1_EPOCHS = args.epochs
    if args.lr:
        opt.LR = args.lr

    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_idx)

    train(opt)
