"""Train CraterInvarianceNet on Multi-Angle / Multi-Rotation Lunar Crater Pairs.

Dataset source: D:/moon/moon (images/train, labels/train, images/val, labels/val)
Extracts authentic crater patches across different rotation & illumination angles
and trains a 128-dimensional invariant metric embedding using AngleInvariantContrastiveLoss.
Accelerated on NVIDIA RTX 3050 GPU (CUDA) or CPU.
"""
import os
import sys
import glob
import re
import argparse
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure app package is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app.pipeline.crater_nn import CraterInvarianceNet, AngleInvariantContrastiveLoss

class LunarCraterPairDataset(Dataset):
    """Dataset generating positive pairs (same crater at 2 rotation angles) and negatives."""

    def __init__(self, data_root: str = "D:/moon/moon", split: str = "train", max_pairs: int = 10000, patch_size: int = 64):
        self.data_root = data_root
        self.split = split
        self.patch_size = patch_size
        self.img_dir = os.path.join(data_root, "images", split)
        self.lbl_dir = os.path.join(data_root, "labels", split)

        print(f"[DATASET] Scanning {self.img_dir} for multi-rotation crater pairs...")
        # Group files by base tile prefix
        # e.g., 'train_CE5_Em2_Craters_TC_120m_Split1_0_0'
        all_imgs = glob.glob(os.path.join(self.img_dir, "*.png"))
        if not all_imgs:
            all_imgs = glob.glob(os.path.join(self.img_dir, "*.jpg"))

        groups = {}
        for img_p in all_imgs:
            fname = os.path.basename(img_p)
            base = re.sub(r'_rotate_\d+\.(png|jpg)', '', fname)
            base = os.path.splitext(base)[0]
            if base not in groups:
                groups[base] = []
            groups[base].append(img_p)

        # Retain groups that have at least 2 rotations
        self.multi_angle_groups = [g for g in groups.values() if len(g) >= 2]
        print(f"[DATASET] Found {len(self.multi_angle_groups)} distinct lunar regions with multi-angle rotations.")
        self.length = min(max_pairs, len(self.multi_angle_groups) * 4)

    def __len__(self):
        return self.length

    def _extract_craters_from_file(self, img_path: str):
        lbl_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        lbl_path = os.path.join(self.lbl_dir, lbl_name)
        craters = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, xc, yc, w, h = map(float, parts[:5])
                        craters.append((xc, yc, w, h))
        return craters

    def __getitem__(self, idx):
        # Pick a random group
        grp = random.choice(self.multi_angle_groups)
        img_a_path, img_b_path = random.sample(grp, 2)

        img_a = cv2.imread(img_a_path, cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(img_b_path, cv2.IMREAD_GRAYSCALE)

        if img_a is None or img_b is None:
            # Fallback black patch
            p_a = np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)
            return p_a, p_a, p_a

        h, w = img_a.shape[:2]
        craters_a = self._extract_craters_from_file(img_a_path)

        if craters_a:
            xc_n, yc_n, bw_n, bh_n = random.choice(craters_a)
            cx, cy = int(xc_n * w), int(yc_n * h)
            rad = max(8, int(max(bw_n, bh_n) * max(w, h) * 0.6))
        else:
            cx, cy = random.randint(32, w - 32), random.randint(32, h - 32)
            rad = 32

        # Extract patch A
        x1, y1 = max(0, cx - rad), max(0, cy - rad)
        x2, y2 = min(w, cx + rad), min(h, cy + rad)
        crop_a = img_a[y1:y2, x1:x2]
        if crop_a.size == 0:
            crop_a = img_a[:self.patch_size, :self.patch_size]
        patch_a = cv2.resize(crop_a, (self.patch_size, self.patch_size)).astype(np.float32) / 255.0

        # Extract patch B from corresponding rotation (or centered region)
        craters_b = self._extract_craters_from_file(img_b_path)
        if craters_b:
            xc_b, yc_b, bw_b, bh_b = random.choice(craters_b)
            cxb, cyb = int(xc_b * w), int(yc_b * h)
            rad_b = max(8, int(max(bw_b, bh_b) * max(w, h) * 0.6))
        else:
            cxb, cyb, rad_b = cx, cy, rad

        x1b, y1b = max(0, cxb - rad_b), max(0, cyb - rad_b)
        x2b, y2b = min(w, cxb + rad_b), min(h, cyb + rad_b)
        crop_b = img_b[y1b:y2b, x1b:x2b]
        if crop_b.size == 0:
            crop_b = img_b[:self.patch_size, :self.patch_size]
        patch_b = cv2.resize(crop_b, (self.patch_size, self.patch_size)).astype(np.float32) / 255.0

        # Sample Negative Patch from non-crater flat plains or another random group
        neg_grp = random.choice(self.multi_angle_groups)
        neg_img_p = random.choice(neg_grp)
        neg_img = cv2.imread(neg_img_p, cv2.IMREAD_GRAYSCALE)
        if neg_img is not None:
            nh, nw = neg_img.shape[:2]
            nx = random.randint(0, max(0, nw - self.patch_size))
            ny = random.randint(0, max(0, nh - self.patch_size))
            crop_neg = neg_img[ny:ny + self.patch_size, nx:nx + self.patch_size]
            patch_neg = cv2.resize(crop_neg, (self.patch_size, self.patch_size)).astype(np.float32) / 255.0
        else:
            patch_neg = np.random.uniform(0.3, 0.7, (self.patch_size, self.patch_size)).astype(np.float32)

        return (
            torch.from_numpy(patch_a).unsqueeze(0),
            torch.from_numpy(patch_b).unsqueeze(0),
            torch.from_numpy(patch_neg).unsqueeze(0)
        )

def main():
    parser = argparse.ArgumentParser(description="Train CraterInvarianceNet on multi-angle crater pairs")
    parser.add_argument("--data", type=str, default="D:/moon/moon", help="Root folder of moon dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--pairs-per-epoch", type=int, default=5000, help="Pairs sampled per epoch")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--output", type=str, default="app/models/crater_invariance_best.pt", help="Checkpoint save path")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("🧠 CRATER INVARIANCE NEURAL NETWORK TRAINING")
    print("=" * 60)
    print(f"  Dataset Root    : {args.data}")
    print(f"  Target Device   : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch Size      : {args.batch_size}")
    print(f"  Pairs/Epoch     : {args.pairs_per_epoch}")
    print(f"  Learning Rate   : {args.lr}")
    print(f"  Output Model    : {args.output}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Initialize Dataset & DataLoader
    train_dataset = LunarCraterPairDataset(data_root=args.data, split="train", max_pairs=args.pairs_per_epoch)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    # Model, Loss, Optimizer
    model = CraterInvarianceNet(embedding_dim=128).to(device)
    loss_fn = AngleInvariantContrastiveLoss(margin=1.0, alpha_pos=1.0, alpha_neg=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_loss = float("inf")
    print("\n[INFO] Starting parameter optimization loop...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        pos_cosines = []
        neg_cosines = []

        for p_a, p_b, p_neg in train_loader:
            p_a = p_a.to(device)
            p_b = p_b.to(device)
            p_neg = p_neg.to(device)

            optimizer.zero_grad()
            z_a = model(p_a)
            z_b = model(p_b)
            z_neg = model(p_neg)

            loss, metrics = loss_fn(z_a, z_b, z_neg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_losses.append(metrics["total_loss"])
            pos_cosines.append(metrics["avg_positive_cosine"])
            neg_cosines.append(metrics["avg_negative_cosine"])

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        avg_pos_cos = np.mean(pos_cosines)
        avg_neg_cos = np.mean(neg_cosines)

        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] | Loss: {avg_loss:.4f} | Pos Cosine: {avg_pos_cos:.4f} | Neg Cosine: {avg_neg_cos:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "pos_cosine": avg_pos_cos,
                "neg_cosine": avg_neg_cos,
                "embedding_dim": 128
            }, args.output)
            print(f"    --> Saved new best checkpoint to {args.output} (Loss: {best_loss:.4f})")

    print("\n" + "=" * 60)
    print("✨ CRATER INVARIANCE TRAINING COMPLETE!")
    print(f"  Final Best Checkpoint: {os.path.abspath(args.output)}")
    print(f"  Best Loss: {best_loss:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
