import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data.load_data import CHARS
from model.LPRNet import build_lprnet

"""文件使用方法:
识别单个文件:
python .\recognize_plate.py `
  --image "E:\temp_code\python_Files\现代电子系统设计课设\licence_plate_imgsets\2026车牌测试集(140张)\1.jpg" `
  --weights "E:\temp_code\python_Files\现代电子系统设计课设\LPRNet_Pytorch-master\weights\run_8\LPRNet__iteration_10000.pth"

  识别整个目录:
python .\recognize_plate.py `
  --image "E:\temp_code\python_Files\现代电子系统设计课设\licence_plate_imgsets\2026车牌测试集(140张)" `
  --weights "E:\temp_code\python_Files\现代电子系统设计课设\LPRNet_Pytorch-master\weights\run_8\LPRNet__iteration_10000.pth"
"""

def decode(logits):
    indices = torch.argmax(logits, dim=1)[0].cpu().numpy().tolist()
    blank = len(CHARS) - 1
    result = []
    previous = None
    for index in indices:
        if index != blank and index != previous:
            result.append(CHARS[index])
        previous = index
    return "".join(result)


def recognize(image_path, model, device):
    encoded_image = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, (94, 24)).astype(np.float32)
    image = (image - 127.5) * 0.0078125
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor)
    return decode(prediction)


def main():
    parser = argparse.ArgumentParser(description="Recognize cropped license plate images with LPRNet")
    parser.add_argument("--image", required=True, help="One image or a directory of cropped plate images")
    parser.add_argument("--weights", default="./weights/Final_LPRNet_model.pth")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA when available")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    model = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_path = Path(args.image)
    image_paths = sorted(input_path.glob("*.jpg")) if input_path.is_dir() else [input_path]
    if not image_paths:
        raise FileNotFoundError(f"No jpg images found in: {input_path}")

    for image_path in image_paths:
        print(f"{image_path.name}: {recognize(image_path, model, device)}")


if __name__ == "__main__":
    main()