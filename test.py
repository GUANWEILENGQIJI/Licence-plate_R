"""Run license-plate detection with YOLO and recognition with LPRNet."""
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
LPRNET_DIR = PROJECT_DIR / "LPRNet_Pytorch-master"
sys.path.insert(0, str(LPRNET_DIR))
from data.load_data import CHARS
from model.LPRNet import build_lprnet

YOLO_MODEL_PATH = PROJECT_DIR / "runs/detect/runs/detect/dataset1/weights/best.pt"
LPRNET_MODEL_PATH = LPRNET_DIR / "weights/run_7/Final_LPRNet_model.pth"
TEST_IMAGE = PROJECT_DIR / "datasets/dataset1/test/images/41_0470.jpg"

def decode_prediction(logits):
    indexes = torch.argmax(logits, dim=1)[0].cpu().numpy().tolist()
    blank = len(CHARS) - 1
    result, previous = [], None
    for index in indexes:
        if index != blank and index != previous:
            result.append(CHARS[index])
        previous = index
    return "".join(result)

def recognize_plate(plate_crop, model, device):
    image = cv2.resize(plate_crop, (94, 24)).astype(np.float32)
    image = np.transpose((image - 127.5) * 0.0078125, (2, 0, 1))
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return decode_prediction(model(tensor))

def main():
    for path, description in ((YOLO_MODEL_PATH, "YOLO model"), (LPRNET_MODEL_PATH, "LPRNet model"), (TEST_IMAGE, "test image")):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(YOLO_MODEL_PATH))
    recognizer = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    recognizer.load_state_dict(torch.load(LPRNET_MODEL_PATH, map_location=device))
    recognizer.to(device).eval()
    results = detector.predict(source=str(TEST_IMAGE), device=0 if device.type == "cuda" else "cpu", conf=0.25, save=False, save_txt=True)
    output_path = PROJECT_DIR / "runs/predict/annotated.jpg"
    crop_dir = PROJECT_DIR / "runs/predict/plates"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    original_image = cv2.imread(str(TEST_IMAGE))
    if original_image is None:
        raise FileNotFoundError(f"Unable to read test image: {TEST_IMAGE}")
    for result in results:
        cv2.imwrite(str(output_path), result.plot())
        for detection_index, box in enumerate(result.boxes):
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            x1, y1 = max(0, min(x1, original_image.shape[1])), max(0, min(y1, original_image.shape[0]))
            x2, y2 = max(0, min(x2, original_image.shape[1])), max(0, min(y2, original_image.shape[0]))
            if x2 <= x1 or y2 <= y1:
                continue
            plate_crop = original_image[y1:y2, x1:x2]
            cv2.imwrite(str(crop_dir / f"plate_{detection_index}.jpg"), plate_crop)
            text = recognize_plate(plate_crop, recognizer, device)
            print(f"车牌框: ({x1}, {y1}, {x2}, {y2})")
            print(f"检测置信度: {confidence:.3f}")
            print(f"LPRNet识别结果: {text or '未识别'}")
    print(f"标注结果已保存到: {output_path.resolve()}")

if __name__ == "__main__":
    main()
