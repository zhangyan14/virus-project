import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from skimage import io

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
        image_path = os.path.join(self.root_dir, self.names_list[index].split(' ')[0])  # 获取图片数据路径
        if not os.path.isfile(image_path):
            print(image_path + ' does not exist!')
            return None
        image = io.imread(image_path)  # 读取图片
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

train_dataset = MyDataset(root_dir='./IMAGEDATA/train', names_file='./IMAGEDATA/train/train.txt', transform=data_transform)

trainset_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0)

# 测试数据加载器
for images, labels in trainset_dataloader:
    print(images.shape, labels)
    break
