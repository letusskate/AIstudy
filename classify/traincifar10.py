import torch
import torchvision
import torchvision.transforms as transforms
import os

############################################### 数据集 ###############################################
transform = transforms.Compose([
    transforms.Resize(224),  # 强制调整图像尺寸至224x224[2,4](@ref)
    transforms.RandomHorizontalFlip(),  # 数据增强：随机水平翻转[1,5](@ref)
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1][1,4](@ref)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 按照ImageNet的均值和标准差进行归一化[2,4](@ref)
])

# 加载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
# testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

################################################ 模型 ################################################
import torchvision.models as models

# 加载预训练模型（自动下载）
model = models.resnet18(pretrained=True)

# 替换最后一层全连接层
num_classes = 10
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# 迁移到GPU（可选）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
##################################################### 训练 #####################################################
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if not os.path.exists('models'):
    os.mkdir('models')

for epoch in range(10):
    ############## train ##############
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if i % 200 == 199:  # 每200批次打印损失
            # print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
        if i % 50 == 49:  # 每50批次打印损失
            print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0

    ############ val ############        
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Val Accuracy: {100 * correct / total:.2f}%')
    torch.save(model,'models/epoch{}.pth'.format(epoch+1)) ## 保存模型
    # torch.save(model.state_dict(), 'models/model.pth') ## 保存模型参数
print('Training Finished')


#%% 若需冻结预训练层（仅训练最后一层），可添加：

# for param in model.parameters():
#     param.requires_grad = False
# model.fc = nn.Linear(model.fc.in_features, 10)  # 仅训练最后一层[6](@ref)