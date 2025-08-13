
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from torchvision.io import read_image
import os

##############################################  数据集 ###############################################
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                images.append((img_path, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        # # 用torchvision.io读取图像（直接返回uint8张量，形状[C,H,W]）
        # image = read_image(img_path)  # 替换Image.open
        # image = image.float() / 255.0  # 替换Image.open # 从[0,255]转为[0,1]的float张量 # 转换数据类型并归一化
        # 用PIL Image读取图像（返回RGB格式的PIL Image）
        image = Image.open(img_path).convert('RGB')
        # 应用预处理流程
        if self.transform:
            image = self.transform(image)
        return image, label


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 数据增强：随机水平翻转[1,5](@ref)
    transforms.ToTensor(),  # 不需要转换为Tensor，因为read_image已经返回了一个张量，加上会报错TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = CustomDataset(data_dir='cifar10data/train', transform=transform)
val_dataset = CustomDataset(data_dir='cifar10data/val', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

########################################### 模型 ###############################################

# 加载预训练的ResNet18模型
model = models.resnet18(pretrained=True)
# 如果需要更高的分类准确率可是使用ResNet50网络模型
# model = models.resnet50(pretrained=True)

# 替换最后的全连接层，classes为分类的类别数。
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

############################################### 训练 ###############################################
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) ## SGD优化器 据我观察效果好

if not os.path.exists('models'):
    os.mkdir('models')

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_loss = 0.0
        for i,(inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_loss += loss.item()
            if i % 50 == 49:  # 每50批次打印损失
                print(f'Epoch [{epoch + 1}, {i + 1}] loss: {batch_loss / 50:.3f}')
                batch_loss = 0.0

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Validation Accuracy: {100 * correct / total}%')
        torch.save(model.state_dict(), f'models/model_epoch{epoch+1}.pth')

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)


# 文件结构：
# my_image_classifier/
# ├── data/
# │   ├── train/
# │   │   ├── class1/
# │   │   ├── class2/
# │   │   └── ...
# │   ├── val/
# │   │   ├── class1/
# │   │   ├── class2/
# │   │   └── ...
# │   └── test/
# │       ├── class1/
# │       ├── class2/
# │       └── ...
# ├── models/
# │   └── model.pth
# └── train.py