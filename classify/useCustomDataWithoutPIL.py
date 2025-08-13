
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.io import read_image
import os


# 加载预训练的ResNet18模型
model = models.resnet18(pretrained=False)

# 替换最后的全连接层，classes为分类的类别数。
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 有10个类别

# 加载模型
model.load_state_dict(torch.load('models/model.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 数据预处理
testtransform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ToTensor(),  # 不需要转换为Tensor，因为read_image已经返回了一个张量，如果加上就会报错TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>
    # transforms.RandomHorizontalFlip(),  # 数据增强：随机水平翻转[1,5](@ref)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 进行预测
def predict(image_path, model, transform):
    # image = Image.open(image_path).convert('RGB') ## 替换Image.open变为torchvision.io.read_image
    # 用torchvision.io读取图像（直接返回uint8张量，形状[C,H,W]）
    image = read_image(image_path)  # 替换Image.open
    # 转换数据类型并归一化
    image = image.float() / 255.0  # 从[0,255]转为[0,1]的float张量
    image = testtransform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# 示例预测
image_path = 'testpic/test_image.jpg'
predicted_class = predict(image_path, model, testtransform)
print(f'Predicted class: {predicted_class}')