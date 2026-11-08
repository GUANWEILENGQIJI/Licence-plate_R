"""

"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO license plate detector")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the custom dataset YAML file",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model or pretrained weights",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--device", default="0", help="CUDA device, for example 0 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Directory for training results",
    )
    parser.add_argument("--name", default="dataset1", help="Training run name")
    parser.add_argument("--resume", action="store_true", help="Resume the last training run")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data.resolve()}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
