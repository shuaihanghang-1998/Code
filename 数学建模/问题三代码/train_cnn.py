# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 15:02
# @Author  : Liu
# @File    : train_cnn.py
# @Software: PyCharm 
# @Comment :

import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


transformer_Image = transforms.Compose([
    transforms.Resize((27, 27)),    # resize图像为27*27，太大的话需要的训练时间太长
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class MyDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


# 划分数据集为训练集和验证集
def split_train_val_data(img_file, label_file, ratio, batch_size, resample=False):
    base_df = pd.read_excel(img_file, sheet_name='training').iloc[:, 1:]
    label_df = pd.read_excel(label_file, sheet_name='training').iloc[:, 1]  # 1 Caco-2
    data = [[] for i in range(2)]  # 2分类
    for idx, row in base_df.iterrows():
        data[label_df[idx]].append(row.tolist())
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    for i, d in enumerate(data):  # d为一个list，保存一个类型的所有图片
        num_sample_train = int(len(d) * ratio[0])
        for x in d[:num_sample_train]:
            train_images.append(x)
            train_labels.append(i)
        for x in d[num_sample_train:]:
            val_images.append(x)
            val_labels.append(i)
    if resample:
        smo = SMOTE(random_state=1)
        train_images, train_labels = smo.fit_resample(train_images, train_labels)
    for i in range(len(train_images)):
        train_images[i] = np.array(train_images[i]).reshape((27, -1))
    for i in range(len(val_images)):
        val_images[i] = np.array(val_images[i]).reshape((27, -1))
    # print(type(train_images), type(train_images[0]))
    train_dataloader = DataLoader(MyDataset(train_images, train_labels, transformer_Image),
                                  batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(MyDataset(val_images, val_labels, transformer_Image),
                                batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


class CNN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=128*4*4, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=n_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class CNN2(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(CNN2, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=24,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=24,
                out_channels=12,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # 第一个全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=12 * 6 * 6, out_features=196),
            nn.ReLU(),
        )
        # 第二个全连接层
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=196, out_features=84),
            nn.ReLU(),
        )
        # 第三个全连接层
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 12*6*6)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.softmax(x, 1)
        return output


# 变量声明
N_CLASSES = 2   # 分类数目
BATCH_SIZE = 32  # 每个batch要放多少张图片
EPOCH = 30
learning_rate = 0.001
img_file = 'data/Molecular_Descriptor.xlsx'
label_file = 'data/ADMET.xlsx'
ratio = [0.8, 0.2]  # 训练集：验证集
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ', DEVICE)


if __name__ == "__main__":
    print(torch.__version__)
    train_loader, val_loader = split_train_val_data(img_file=img_file, label_file=label_file, ratio=ratio, batch_size=BATCH_SIZE, resample=True)
    cnn = CNN2(in_channels=3, n_classes=N_CLASSES)
    cnn.to(DEVICE)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        train_accuracy = []
        for step, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            b_x = Variable(x)
            b_y = Variable(y)
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                val_accuracy = []
                for (data, target) in val_loader:
                    data = data.to(DEVICE)
                    target = target.to(DEVICE)
                    val_output = cnn(data)
                    pred = torch.max(val_output, 1)[1]
                    val_acc = sum(pred == target)
                    val_num = len(target)
                    val_accuracy.append([val_acc, val_num])
                # val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
                val_r = (sum([tup[0] for tup in val_accuracy]), sum([tup[1] for tup in val_accuracy]))
                print('Epoch: ', epoch, '|train loss: %.4f' % loss, '验证集上准确率：', (val_r[0].float()/val_r[1]).to('cpu').numpy())

    # torch.save(cnn, './model/model2.pkl')  # 保存模型
