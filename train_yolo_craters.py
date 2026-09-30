"""Train a YOLO Crater Detection Model on Lunar Imagery.

Supports Ultralytics YOLOv8 / YOLO11 on CUDA (NVIDIA RTX 3050) or CPU.
Dataset configuration: lunar_craters.yaml (pointing to D:/moon/moon).
"""
import os
import sys
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO crater detector on lunar dataset")
    parser.add_argument("--data", type=str, default="lunar_craters.yaml", help="Path to dataset YAML file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model architecture (yolov8n.pt, yolov8s.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (16 recommended for 6GB RTX 3050)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('0' for GPU, 'cpu', or 'auto')")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--name", type=str, default="lunar_crater_run", help="Training experiment name")
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("🌙 LUNAR CRATER YOLO DETECTOR TRAINING")
    print("=" * 60)
    print(f"  Dataset YAML : {os.path.abspath(args.data)}")
    print(f"  Base Model   : {args.model}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch Size   : {args.batch}")
    print(f"  Image Size   : {args.imgsz}")
    print(f"  Device       : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != 'cpu' else 'CPU'})")
    print(f"  CUDA Memory  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() and device != 'cpu' else "  CUDA Memory  : N/A")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Please install it using pip install ultralytics")
        sys.exit(1)

    # Initialize model
    model = YOLO(args.model)

    # Train model
    print(f"\n[INFO] Commencing training on {args.data}...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        name=args.name,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("✨ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"  Best Weights Saved To: {results.save_dir}/weights/best.pt")
    print(f"  Last Weights Saved To: {results.save_dir}/weights/last.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
