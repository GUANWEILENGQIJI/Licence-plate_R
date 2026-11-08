"""YOLO 车牌检测接文字识别

python final_version1.py `
    --input_p "E:\temp_code\python_Files\现代电子系统设计课设\licence_plate_imgsets\2026车牌测试集(140张)\7.jpg" `
    --img_w "E:\temp_code\python_Files\现代电子系统设计课设\runs\detect\runs\detect\dataset1-3\weights\best.pt" `
    --rec_w "E:\temp_code\python_Files\现代电子系统设计课设\LPRNet_Pytorch-master\weights\run_2\Final_LPRNet_model.pth"

python final_version1.py --input_p "E:\temp_code\python_Files\现代电子系统设计课设\licence_plate_imgsets\2026车牌测试集(140张)" `
    --img_w "E:\temp_code\python_Files\现代电子系统设计课设\runs\detect\runs\detect\dataset1-3\weights\best.pt" `
    --rec_w "E:\temp_code\python_Files\现代电子系统设计课设\LPRNet_Pytorch-master\weights\run_4\Final_LPRNet_model.pth"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
LPRNET_DIR = PROJECT_DIR / "LPRNet_Pytorch-master"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Make the bundled LPRNet package importable when this script is run from the project root.
sys.path.insert(0, str(LPRNET_DIR))
from data.load_data import CHARS  # noqa: E402
from model.LPRNet import build_lprnet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect license plates with YOLO and recognize them with LPRNet."
    )
    parser.add_argument(
        "--input_p", required=True,
        help="Input image path or a directory containing images.",
    )
    parser.add_argument(
        "--img_w", required=True,
        help="YOLO license-plate detector weights (.pt).",
    )
    parser.add_argument(
        "--rec_w", required=True,
        help="LPRNet recognition weights (.pth).",
    )
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO detection confidence threshold (default: 0.25).")
    parser.add_argument("--device", default=None,
                        help="Inference device, for example 0 or cpu. Defaults to CUDA when available.")
    return parser.parse_args()


def collect_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {input_path}")
        return [input_path]
    if input_path.is_dir():
        images = sorted(
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No supported images found in: {input_path}")
        return images
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def read_image(path: Path):
    # imdecode supports Unicode paths on Windows more reliably than cv2.imread.
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def decode_prediction(logits):
    indexes = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().tolist()
    blank = len(CHARS) - 1
    decoded = []
    previous = None
    for index in indexes:
        if index != blank and index != previous:
            decoded.append(CHARS[index])
        previous = index
    return "".join(decoded)


def recognize_crop(crop, recognizer, device):
    image = cv2.resize(crop, (94, 24)).astype(np.float32)
    image = (image - 127.5) * 0.0078125
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = recognizer(tensor)
    return decode_prediction(logits)


def load_recognizer(weights_path: Path, device):
    model = build_lprnet(
        lpr_max_len=8,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=0,
    )
    checkpoint = torch.load(str(weights_path), map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    input_path = Path(args.input_p).expanduser()
    detector_weights = Path(args.img_w).expanduser()
    recognizer_weights = Path(args.rec_w).expanduser()

    for path, description in (
        (detector_weights, "YOLO detector weights"),
        (recognizer_weights, "LPRNet recognizer weights"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    images = collect_images(input_path)
    device = torch.device(
        f"cuda:{args.device}" if args.device is not None and args.device.isdigit()
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    ) if args.device != "cpu" else torch.device("cpu")

    detector = YOLO(str(detector_weights))
    recognizer = load_recognizer(recognizer_weights, device)
    yolo_device = args.device if args.device is not None else (0 if device.type == "cuda" else "cpu")

    for image_path in images:
        image = read_image(image_path)
        results = detector.predict(source=image, conf=args.conf, device=yolo_device, verbose=False)
        detections = 0
        print(f"\n图片: {image_path}")
        for result in results:
            for index, box in enumerate(result.boxes):
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, x2 = max(0, min(x1, image.shape[1])), max(0, min(x2, image.shape[1]))
                y1, y2 = max(0, min(y1, image.shape[0])), max(0, min(y2, image.shape[0]))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                text = recognize_crop(crop, recognizer, device)
                confidence = float(box.conf[0])
                print(f"  车牌{index + 1}: {text or '未识别'} | 置信度: {confidence:.4f} | 框: ({x1}, {y1}, {x2}, {y2})")
                detections += 1
        if detections == 0:
            print("  未检测到车牌")


if __name__ == "__main__":
    main()
