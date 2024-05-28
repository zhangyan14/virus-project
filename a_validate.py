import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 定义与训练时相同的模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 加载模型权重
model = SimpleCNN()
model.load_state_dict(torch.load('simple_cnn_epoch_32_loss_0.0319.pth'))
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()  # 切换到评估模式

def read_image_paths(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]


# 预处理函数
def preprocess_images(image_paths):
    images = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        images.append(image)
        
    images = torch.stack(images)  # 组合成一个批次
    return images
# 加载并预处理图像
image_paths = read_image_paths('C:/Users/Peace/workspace/yolo/dataset/val/image_paths.txt')

input_images = preprocess_images(image_paths)
input_images = input_images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 进行预测
with torch.no_grad():  # 禁用梯度计算
    output = model(input_images)
    outputs = model(input_images)
    predictions = torch.round(outputs).squeeze()
print('Prediction:', predictions)