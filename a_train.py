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

# 预处理函数
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 增加批次维度
    return image

# 加载并预处理图像
image_path = 'C:/Users/Peace/workspace/yolo/dataset/val/class1/10.jpg'
input_image = preprocess_image(image_path)
input_image = input_image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 进行预测
with torch.no_grad():  # 禁用梯度计算
    output = model(input_image)
    prediction = torch.round(output)  # 对输出进行四舍五入（因为是二分类问题）
    prediction = prediction.item()  # 将张量转换为Python标量

print('Prediction:', prediction)