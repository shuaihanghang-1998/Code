import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader, Dataset
from torchvision import transforms
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

class MyDataSet(Dataset):
    # train 3500*3*32*32
    # test 700
    def __init__(self, root, partition = 'train'):
        super(MyDataSet, self).__init__()
        self.root = root
        self.partition = partition
        self.imgs = []
        self.labs = []
        img_list = os.listdir(root + partition + '/')
        img_list.sort(key=lambda x: int(x.split('.jpg')[0]))
        for idx in img_list:
            img = io.imread(root + partition + '/' + idx) # 32*32*3
            img = transform.resize(img, (64,64,3))
            img = np.transpose(img, (2,0,1)) # 3*64*64
            self.imgs.append(img)
        with open(root + partition + '_labels.csv') as csvfile:
            reader = csv.reader(csvfile)
            self.labs = [float(row[0]) for row in reader] # 3500 float print(len(self.labs), self.labs[-5 :])
    def __getitem__(self, index):
        img, lab = self.imgs[index], self.labs[index]
        return img, lab
    def __len__(self):
        return len(self.labs)

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCH = 15
BATCH_SIZE = 100
LR = 0.0001
WEGHT_DECAY = 1e-6
NUM_CLASS = 10

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.right = shortcut
    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        out = F.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        # ResNet34开始的部分
        self.pre =nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1))
        # ResNet34 具有重复模块的部分 分别有3，4，6，3个residual block
        self.layer1 = self.make_layer(64,128,3)
        self.layer2 = self.make_layer(128,256,4,2)
        self.layer3 = self.make_layer(256,512,6,2)
        self.layer4 = self.make_layer(512,512,3,2)
        # ResNet  末尾的部分，分类用的全连接
        self.fc = nn.Linear(512,NUM_CLASS) 
    def make_layer(self, in_channels, out_channels, block_num, stride=1):
        #构建一个layer，包含多个residual block
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 2)#kernel_size为7是因为经过多次下采样之后feature map的大小为7*7，即224->112->56->28->14->7
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

def train(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEGHT_DECAY)
    for epoch in range(EPOCH):
        train_loss = 0.0
        best_loss = 100.0
        train_acc = 0.0
        for i, data in enumerate(train_loader):
            inputs, targets = data #inputs <class 'torch.Tensor'> torch.Size([100, 3, 64, 64])    targets <class 'torch.Tensor'> torch.Size([100])
            inputs = inputs.float()
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(input=outputs, dim=1)
            train_acc += sum((predicted == targets).float()) / len(targets)
            if(i + 1) % 10 == 0:
                print('Eopch {},  Loss: {:.4f}, Acc: {:.4f}'.
                format(epoch+1,  train_loss/10 , train_acc/10))
                if(best_loss > train_loss):
                    best_loss = train_loss
                    torch.save(model.state_dict(), './trained_cnn_model.pth') # model.state_dict()只保存模型的参数， model保存整个模型
                train_loss = 0.0
                train_acc =0.0
    print('Finish training')

def test(model, test_loader):
    test_acc = 0.0
    for inputs, targets in test_loader:
        inputs = inputs.float()
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        predicted = torch.argmax(input=outputs, dim=1)
        test_acc += sum((predicted == targets).float()) / len(targets)
    print('Val Acc: {:.4f}'.format(test_acc/len(test_loader)))

if __name__ == '__main__':
    train_data = MyDataSet(root='D:/data/StreetNumbers/', partition='train')
    test_data = MyDataSet(root='D:/data/StreetNumbers/', partition='test')
    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=BATCH_SIZE)

    model = ResNet34()
    model.to(DEVICE)
    train(model, train_loader)
    test(model, test_loader)

    # print(model)
    # input = torch.randn(10,3,64,64)
    # input = input.to(DEVICE)
    # out = model(input)
    # print(out)

    # for i in range(10):
    #     img, lb = train_data.__getitem__(i)
    #     print(type(img), img.shape, lb)
    #     ax = plt.subplot(2,5,i + 1)
    #     ax.imshow(img)
    # plt.pause(10)
