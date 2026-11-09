1.# 车牌识别系统项目说明

本项目面向中国车牌场景，采用“目标检测 + 车牌字符识别”的两阶段方案，实现车牌定位和内容识别。核心思路包括：

- 使用 YOLOv8 对车牌区域进行检测；
- 使用 LPRNet 对检测到的车牌图像进行字符识别；
- 针对少样本省份/地区车牌（如甘、桂等）进行数据扩充与训练优化；
- 通过调整网络通道数和增强训练数据提升识别稳定性。

该项目结合 CCPD、CRPD 及补充的少样本车牌数据集进行训练，最终在 CLPD 数据集和 140 张测试图片上达到预期的识别准确率。

## 1. 项目背景

车牌识别（License Plate Recognition, LPR）在智能交通、停车场管理、门禁监控等场景中具有广泛需求。由于不同地区车牌字符分布不均衡，少样本类别（如“甘”“桂”等）容易出现识别错误，导致识别率下降。

本项目从数据分布出发，重点优化少样本类别的识别能力，并在保持模型轻量化的前提下提升准确率。

## 2. 技术方案

### 2.1 检测模块

- 模型：YOLO
- 作用：从原始图像中定位车牌区域
- 代码入口：`train.py`、`final_version1.py`

### 2.2 识别模块

- 模型：LPRNet
- 作用：对裁剪后的车牌区域进行字符识别
- 代码目录：`LPRNet_Pytorch-master/`

### 2.3 数据增强与优化

训练初期，模型识别效果不理想，主要错误集中在样本量较少的车牌类别上。例如：

- 甘
- 桂
- 其他少样本省份前缀

为改善这一问题，项目采取了以下策略：

1. 搜集并扩充少样本车牌数据集；
2. 对长尾类别进行补充训练；
3. 增强特征提取能力，优化网络结构；
4. 平衡识别率和显存占用及训练时间。

## 3. 训练过程与改进

训练过程中，项目对网络结构进行了迭代优化，重点测试了通道数扩展带来的效果：

- 当前网络通道：64 → 128 → 256 → 256
- 扩充后网络通道：96 → 192 → 384 → 384

该调整显著增强了网络特征提取能力，使得模型对复杂车牌样本和少样本类别的识别能力更强；同时也带来了更高的显存消耗和更长的训练时间。

最终，在 CLPD 数据集和 140 张测试图片上的识别结果达到预期要求，整体识别效果稳定可用。

## 4. 项目目录结构

```text
.
├── LPRNet_Pytorch-master/        # LPRNet 识别模型代码
│   ├── data/                    # 数据集与处理脚本
│   ├── model/                   # 模型定义
│   ├── weights/                 # 训练权重
│   ├── train_LPRNet.py          # LPRNet 训练脚本
│   ├── test_LPRNet.py           # LPRNet 测试脚本
│   ├── predict_LRPNet.py        # 单张车牌预测脚本
│   └── README.md                # 子项目说明
├── runs/                        # YOLO 检测训练结果
├── final_version1.py            # 端到端预测脚本（检测 + 识别）
├── train.py                     # YOLO 训练脚本
├── toLPRNet_data.py             # 车牌裁剪与数据生成脚本
├── settings.py                  # 配置或实验参数
├── Readme.md                    # 项目总说明
├── todolist.txt                 # 任务清单
├── .gitignore
└── LICENSE
```

## 5. 环境依赖

建议使用 Python 3.9+ / 3.10+，并安装以下依赖：

```bash
pip install torch torchvision
pip install ultralytics opencv-python numpy pillow imutils
```

如果使用 CUDA 训练或推理，请确保本机已正确安装 NVIDIA 驱动和对应版本的 PyTorch。

## 6. 快速使用

### 6.1 训练 YOLO 检测模型

```bash
python train.py --data your_dataset.yaml --model yolov8n.pt --epochs 100 --batch 16 --imgsz 640 --device 0
```

参数说明：

- `--data`：自定义数据集配置文件（YAML）
- `--model`：YOLO 预训练权重
- `--epochs`：训练轮数
- `--batch`：批大小
- `--imgsz`：输入图像尺寸
- `--device`：训练设备，如 `0` 或 `cpu`

### 6.2 运行端到端识别

```bash
python final_version1.py \
  --input_p "E:\path\to\image_or_folder" \
  --img_w "E:\path\to\best.pt" \
  --rec_w "E:\path\to\Final_LPRNet_model.pth"
```

示例：

```bash
python final_version1.py --input_p "E:\temp_code\python_Files\现代电子系统设计课设\licence_plate_imgsets\2026车牌测试集(140张)" --img_w "E:\temp_code\python_Files\现代电子系统设计课设\runs\detect\runs\detect\dataset1-3\weights\best.pt" --rec_w "E:\temp_code\python_Files\现代电子系统设计课设\LPRNet_Pytorch-master\weights\run_4\Final_LPRNet_model.pth"
```

### 6.3 识别单张车牌

如果只需要对单张车牌进行识别，可使用 LPRNet 自带的预测脚本：

```bash
cd LPRNet_Pytorch-master
python predict_LRPNet.py --image "path/to/plate.jpg" --pretrained_model "weights/Final_LPRNet_model.pth"
```

## 7. 训练结果总结

本项目的最终目标是兼顾识别准确率与工程实用性，经过以下改进后达到目标：

- 补充少样本车牌数据，改善长尾类别识别偏差；
- 扩大网络通道数，提高特征表达能力；
- 使用 YOLO 检测 + LPRNet 识别的两阶段结构提升整体稳定性；
- 在 CLPD 与外部测试集上取得符合课程要求的识别效果。

## 8. 说明与后续改进方向

后续如果继续优化该项目，可以考虑：

- 扩充更多地区车牌样本，提升模型泛化能力；
- 加入数据增强策略，如旋转、遮挡、光照变化模拟；
- 进一步尝试更深或更轻量的检测模型；
- 对车辆与车牌检测结果进行去重和后处理，提高稳定性。

## 9. 参考资料

- LPRNet: License Plate Recognition via Deep Neural Networks
- Ultralytics YOLO 官方文档
- CCPD / CRPD 等车牌公开数据集

## 10. 结语

本项目实现了一个基于深度学习的车牌检测与识别系统，能够在真实场景中完成车牌定定位和字符识别任务，并通过长尾数据补充和网络结构优化，解决了少样本类别识别错误的问题。

如果需要进一步开发或部署成可视化 Web 应用/接口服务，可在现有结构基础上继续扩展。
