"""predict a license plate"""
"""使用方法:
python predict_LRPNet.py --...

参数说明：
--image              要识别的图片路径，必填
--pretrained_model   模型权重路径，默认是 weights/Final_LPRNet_model.pth
--img_size WIDTH HEIGHT
                      默认是 94 24
--lpr_max_len        车牌最大长度，默认是 8
--dropout_rate       默认是 0
--cuda               使用 CUDA,没有此参数时使用 CPU
"""

import argparse
import os

import cv2
import numpy as np
import torch

from data.load_data import CHARS
from model.LPRNet import build_lprnet


def get_parser():
    parser = argparse.ArgumentParser(
        description="Predict a license plate number from one image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input license plate image.",
    )
    parser.add_argument(
        "--pretrained_model",
        default="./weights/Final_LPRNet_model.pth",
        help="Path to the trained LPRNet weights.",
    )
    parser.add_argument(
        "--img_size",
        nargs=2,
        type=int,
        default=[94, 24],
        metavar=("WIDTH", "HEIGHT"),
        help="Model input size (default: 94 24).",
    )
    parser.add_argument(
        "--lpr_max_len",
        type=int,
        default=8,
        help="Maximum license plate length used by the model.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Dropout rate used when constructing the model.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run inference on CUDA when CUDA is available.",
    )
    return parser


def load_image(image_path, image_size):
    """Read and normalize an image using the same preprocessing as training."""
    encoded_image = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to read image: {}".format(image_path))

    width, height = image_size
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height))

    image = image.astype(np.float32)
    image -= 127.5
    image *= 0.0078125
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def decode_prediction(logits):
    """Decode LPRNet logits by removing repeated characters and blank tokens."""
    prediction = logits.detach().cpu().numpy()[0]
    blank_index = len(CHARS) - 1
    indexes = np.argmax(prediction, axis=0)

    decoded = []
    previous = blank_index
    for index in indexes:
        if index == blank_index or index == previous:
            previous = index
            continue
        decoded.append(CHARS[index])
        previous = index
    return "".join(decoded)


def predict_image(model, image_path, image_size, device):
    """Return the plate number predicted for an image path."""
    image = load_image(image_path, image_size).to(device)
    with torch.no_grad():
        logits = model(image)
    return decode_prediction(logits)


def main():
    args = get_parser().parse_args()
    if not os.path.isfile(args.image):
        raise FileNotFoundError("Input image does not exist: {}".format(args.image))
    if not os.path.isfile(args.pretrained_model):
        raise FileNotFoundError(
            "Pretrained model does not exist: {}".format(args.pretrained_model)
        )

    use_cuda = args.cuda and torch.cuda.is_available()
    if args.cuda and not use_cuda:
        print("[Warning] CUDA is not available; using CPU instead.")
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = build_lprnet(
        lpr_max_len=args.lpr_max_len,
        phase=False,
        class_num=len(CHARS),
        dropout_rate=args.dropout_rate,
    )
    checkpoint = torch.load(args.pretrained_model, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    plate_number = predict_image(model, args.image, args.img_size, device)
    print("Predict: {}".format(plate_number))


if __name__ == "__main__":
    main()