import os

import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.color import gray2rgb
import torch.nn.functional as F

class MyDataset(Dataset):
    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir  # 根目录
        self.names_file = names_file  # .txt文件路径
        self.transform = transform  # 数据预处理
        self.size = 0  # 数据集大小
        self.names_list = []  # 数据集路径列表

        if not os.path.isfile(self.names_file):
            print(self.names_file + ' does not exist!')
            return
        
        with open(self.names_file) as file:
            for f in file:  # 循环读取.txt文件总每行数据信息
                self.names_list.append(f.strip())  # 去除换行符
                self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        
        image_path = os.path.join(self.root_dir, self.names_list[index].split(' ')[0])
        if not os.path.isfile(image_path):
            print(image_path + ' does not exist!')
            return None
        image = io.imread(image_path)  # 读取图片

        if len(image.shape) == 2:  # 灰度图像
            image = gray2rgb(image)
        label = int(self.names_list[index].split(' ')[1])  # 读取标签

        if self.transform:
            image = self.transform(image)  # 应用预处理

        return image, label

# 定义数据预处理
data_transform = transforms.Compose([
    transforms.ToPILImage(),  # 将numpy数组转为PIL图像
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将PIL图像转为Tensor
])


train_dataset = MyDataset(root_dir='C:/Users/Peace/workspace/yolo/dataset', names_file='C:/Users/Peace/workspace/yolo/dataset/train/train.txt', transform=data_transform)

trainset_dataloader = DataLoader(dataset=train_dataset, batch_size=1,)

# 测试数据加载器
for images, labels in trainset_dataloader:
    print(images.shape, labels)
    break



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)  # 二分类问题，输出为1

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积层1 + ReLU + 池化层

        x = self.pool(F.relu(self.conv2(x)))  # 卷积层2 + ReLU + 池化层
        x = x.view(-1, 32 * 56 * 56)          # 展平张量
        x = F.relu(self.fc1(x))               # 全连接层1 + ReLU
        x = torch.sigmoid(self.fc2(x))        # 全连接层2 + Sigmoid激活函数
        return x

model = SimpleCNN()
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(50):
    running_loss = 0.0
    for images, labels in trainset_dataloader:
        images, labels = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).float().unsqueeze(1)  # 将标签转换为float并增加维度
        
        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新参数


        optimizer.step()
        
        # 记录损失
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/50], Loss: {running_loss/len(trainset_dataloader):.4f}')
     # Save the model if loss is less than 0.1
    if running_loss < 0.1:
        torch.save(model.state_dict(), f'simple_cnn_epoch_{epoch+1}_loss_{running_loss:.4f}.pth')
        print(f'Saved model at epoch {epoch+1} with loss {running_loss:.4f}')
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             print(f'Layer: {name} | Size: {param.size()} | Values : {param[:2]}')  # 打印前两个值作为示例
# torch.save(model.state_dict(), 'simple_cnn1.pth')
print('Finished Training')




