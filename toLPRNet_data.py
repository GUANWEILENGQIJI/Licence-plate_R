"""Detect CLPD plates, crop them, and name crops with labels from CLPD.csv."""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = Path(
    r"E:\temp_code\python_Files\现代电子系统设计课设\licence_plate_imgsets\CLPD\CLPD_1200"
)
DEFAULT_MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "dataset2"
    / "weights"
    / "best.pt"
)
DEFAULT_CSV_PATH = PROJECT_DIR / "licence_plate_imgsets" / "CLPD" / "CLPD.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "LPRNet_Pytorch-master" / "data" / "test1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and crop plates, then name them with CLPD.csv labels."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--csv", dest="csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_labels(csv_path):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            return [row["label"].strip() for row in rows]
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unable to decode CSV: {csv_path}")


def image_index(path):
    match = re.fullmatch(r"(?:plate_)?(\d+)", path.stem, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Expected a numeric file name such as 0.jpg: {path.name}")
    return int(match.group(1))


def safe_label(label):
    label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", label).strip()
    if not label:
        raise ValueError("CSV contains an empty plate label")
    return label


def read_image(path):
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def write_image(path, image):
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"Unable to encode image: {path}")
    encoded.tofile(path)


def get_output_path(output_dir, label, suffix, overwrite, detection_count):
    if overwrite and detection_count == 0:
        return output_dir / f"{label}{suffix}"

    target = output_dir / f"{label}{suffix}"
    if not target.exists() and detection_count == 0:
        return target

    count = detection_count + 1
    while True:
        target = output_dir / f"{label}_{count}{suffix}"
        if not target.exists():
            return target
        count += 1


def main():
    args = parse_args()
    for path, description in (
        (args.source_dir, "source directory"),
        (args.model, "model"),
        (args.csv_path, "CSV file"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    labels = read_labels(args.csv_path)
    image_paths = sorted(
        (
            path
            for path in args.source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=image_index,
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    processed = 0
    cropped = 0

    for image_path in image_paths:
        index = image_index(image_path)
        if index >= len(labels):
            raise IndexError(
                f"{image_path.name} needs CSV row {index + 1}, "
                f"but CSV contains only {len(labels)} data rows"
            )

        image = read_image(image_path)
        results = model.predict(
            source=image,
            device=args.device,
            conf=args.conf,
            verbose=False,
        )
        label = safe_label(labels[index])
        detection_count = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                x1 = max(0, min(x1, image.shape[1]))
                y1 = max(0, min(y1, image.shape[0]))
                x2 = max(0, min(x2, image.shape[1]))
                y2 = max(0, min(y2, image.shape[0]))
                if x2 <= x1 or y2 <= y1:
                    continue

                plate_crop = image[y1:y2, x1:x2]
                target = get_output_path(
                    args.output_dir, label, ".jpg", args.overwrite, detection_count
                )
                write_image(target, plate_crop)
                print(f"{image_path.name} -> {target.name}")
                detection_count += 1
                cropped += 1

        if detection_count == 0:
            print(f"{image_path.name}: no plate detected")
        processed += 1

    print(f"Processed: {processed}, cropped: {cropped}")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
