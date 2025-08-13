
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.io import read_image
import os

modelpath = 'models/epoch2.pth'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ToTensor(), ## ds说，不用pil，就不需要totensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(modelpath)
model = model.to(device)
model.eval()

# 进行预测
def predict(image_path, model, transform):
    # 用torchvision.io读取图像（直接返回uint8张量，形状[C,H,W]）
    image = read_image(image_path)  # 替换Image.open
    # 转换数据类型并归一化
    image = image.float() / 255.0  # 从[0,255]转为[0,1]的float张量
    # 应用预处理流程
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# 示例预测
image_path = 'testpic/test_image2.jpg'
predicted_class = predict(image_path, model, transform)
print(f'Predicted class: {predicted_class}')