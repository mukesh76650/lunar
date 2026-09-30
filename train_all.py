"""Dual-Model Lunar Crater Training Pipeline.

Trains both:
1. YOLO Crater Detector (Bounding Box Crater Object Detection)
2. CraterInvarianceNet (Multi-Angle Deep Invariant Embedding Network)
Targeting the dataset at D:/moon/moon on NVIDIA RTX 3050 GPU (CUDA).
"""
import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Train both Crater Invariance and YOLO models on D:/moon")
    parser.add_argument("--data", type=str, default="D:/moon/moon", help="Root directory of moon dataset")
    parser.add_argument("--epochs-yolo", type=int, default=25, help="Number of YOLO epochs")
    parser.add_argument("--epochs-invariance", type=int, default=15, help="Number of CraterInvarianceNet epochs")
    parser.add_argument("--batch-yolo", type=int, default=16, help="YOLO batch size")
    parser.add_argument("--batch-invariance", type=int, default=64, help="Invariance Net batch size")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO training")
    parser.add_argument("--skip-invariance", action="store_true", help="Skip CraterInvarianceNet training")
    args = parser.parse_args()

    python_exe = sys.executable

    print("\n" + "=" * 65)
    print("🌕 COMPLETE LUNAR CRATER DUAL TRAINING PIPELINE")
    print("=" * 65)
    print(f"  Python Interpreter : {python_exe}")
    print(f"  Dataset Location   : {os.path.abspath(args.data)}")
    print(f"  YOLO Training      : {'Enabled (' + str(args.epochs_yolo) + ' epochs)' if not args.skip_yolo else 'Skipped'}")
    print(f"  Invariance Training: {'Enabled (' + str(args.epochs_invariance) + ' epochs)' if not args.skip_invariance else 'Skipped'}")
    print("=" * 65 + "\n")

    # 1. Train CraterInvarianceNet
    if not args.skip_invariance:
        print("\n" + "-" * 65)
        print("STAGE 1/2: Training CraterInvarianceNet (Multi-Angle Contrastive Loss)")
        print("-" * 65)
        cmd_inv = [
            python_exe,
            "train_crater_invariance.py",
            "--data", args.data,
            "--epochs", str(args.epochs_invariance),
            "--batch-size", str(args.batch_invariance),
            "--output", "app/models/crater_invariance_best.pt"
        ]
        ret = subprocess.run(cmd_inv)
        if ret.returncode != 0:
            print("[ERROR] CraterInvarianceNet training encountered an error.")
        else:
            print("[SUCCESS] Stage 1 Completed. Model saved to app/models/crater_invariance_best.pt")

    # 2. Train YOLO Crater Detector
    if not args.skip_yolo:
        print("\n" + "-" * 65)
        print("STAGE 2/2: Training YOLO Crater Detector (Ultralytics YOLOv8)")
        print("-" * 65)
        cmd_yolo = [
            python_exe,
            "train_yolo_craters.py",
            "--data", "lunar_craters.yaml",
            "--epochs", str(args.epochs_yolo),
            "--batch", str(args.batch_yolo),
            "--name", "lunar_crater_run"
        ]
        ret = subprocess.run(cmd_yolo)
        if ret.returncode != 0:
            print("[ERROR] YOLO training encountered an error.")
        else:
            print("[SUCCESS] Stage 2 Completed. Weights saved to runs/detect/lunar_crater_run/weights/best.pt")

    print("\n" + "=" * 65)
    print("🎉 ALL TRAINING STAGES COMPLETE!")
    print("=" * 65)

if __name__ == "__main__":
    main()
