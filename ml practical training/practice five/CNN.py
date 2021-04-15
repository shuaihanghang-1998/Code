import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import zipfile   
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
        
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
transformer_Image = transforms.Compose([
    transforms.Resize((128,128)),    # resize图像为28*28，太大的话需要的训练时间太长
    transforms.ToTensor(),
    normalize
])
   
class FlowerDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
     
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]
         
# 划分数据集为训练集和验证集
def split_train_val_data(data_dir, ratio, batch_size):
    # the sum of ratio must equal to 1
    file = zipfile.ZipFile('D:/data/flowers.zip', 'r')
    #file.extractall('D:/data/flowers/')
    #print("&&&&&",'解压到D:/data/flowers')
    dataset = ImageFolder('D:/data/flowers/'+file.namelist()[0])  # data_dir精确到分类目录的上一级
    character = [[] for i in range(len(dataset.classes))]
    # print(dataset.class_to_idx)
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)
    print(type(character[0][0]))
    # print(dataset.samples)
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
   
    for i, data in enumerate(character):  # data为一类图片
        num_sample_train = int(len(data) * ratio[0])
        # print(num_sample_train)
        for x in data[:num_sample_train]:
            train_images.append(x)
            train_labels.append(i)
        for x in data[num_sample_train:]:
            val_images.append(x)
            val_labels.append(i)
    # print(len(train_inputs))
    train_dataloader = DataLoader(FlowerDataset(train_images, train_labels, transformer_Image),
                                  batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(FlowerDataset(val_images, val_labels, transformer_Image),
                                batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader

class CNN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(CNN, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=20,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=20 * 7 * 7, out_features=98),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=98, out_features=n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        
        x = self.conv1(x)
        _, axes = plt.subplots(nrows=1, ncols=8, figsize=(10, 3))
        i=0
        x_show=x.detach().numpy()
        for ax in axes :
            ax.set_axis_off()
            ax.imshow(x_show[0,i,:,:], cmap=plt.cm.gray_r, interpolation='nearest')
            i=i+1
        plt.show()
        x = self.conv2(x)
        _, axes = plt.subplots(nrows=1, ncols=12, figsize=(10, 3))
        i=0
        x_show=x.detach().numpy()
        for ax in axes :
            ax.set_axis_off()
            ax.imshow(x_show[0,i,:,:], cmap=plt.cm.gray_r, interpolation='nearest')
            i=i+1
        plt.show()
        x = self.conv3(x)
        _, axes = plt.subplots(nrows=1, ncols=16, figsize=(10, 3))
        i=0
        x_show=x.detach().numpy()
        for ax in axes :
            ax.set_axis_off()
            ax.imshow(x_show[0,i,:,:], cmap=plt.cm.gray_r, interpolation='nearest')
            i=i+1
        plt.show()
        x = self.conv4(x)
        _, axes = plt.subplots(nrows=1, ncols=20, figsize=(10, 3))
        i=0
        x_show=x.detach().numpy()
        for ax in axes :
            ax.set_axis_off()
            ax.imshow(x_show[0,i,:,:], cmap=plt.cm.gray_r, interpolation='nearest')
            i=i+1
        plt.show()
        
        #print("x.shape", x.shape)
        x = x.view(-1, 20 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.softmax(x, 1)
        return output

# 变量声明
N_CLASSES = 5   #花卉种类数
BATCH_SIZE = 8  # 每个batch要放多少张图片
EPOCH = 10
learning_rate = 0.005
data_dir = "D:/data/flowers"
ratio = [0.8, 0.2]  #训练集：验证集

if __name__ == "__main__":
    print(torch.__version__)

    train_loader, val_loader = split_train_val_data(data_dir='D:/data/flowers', ratio=[0.8, 0.2], batch_size=BATCH_SIZE)

    cnn = CNN(in_channels=3, n_classes=N_CLASSES)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        tra_accuracy = []
        for step, (x, y) in enumerate(train_loader):
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
                    val_output = cnn(data)
                    pred = torch.max(val_output, 1)[1]
                    val_acc = sum(pred == target)
                    val_num = len(target)
                    val_accuracy.append([val_acc, val_num])
                # val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
                val_r = (sum([tup[0] for tup in val_accuracy]), sum([tup[1] for tup in val_accuracy]))
                for (data, target) in train_loader:
                    tra_output = cnn(data)
                    pred = torch.max(tra_output, 1)[1]
                    tra_acc = sum(pred == target)
                    tra_num = len(target)
                    tra_accuracy.append([tra_acc, tra_num])
                # val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
                tra_r = (sum([tup[0] for tup in tra_accuracy]), sum([tup[1] for tup in tra_accuracy]))
                print('Epoch: ', epoch, '|train loss: %.4f' % loss, '训练集上准确率：', (tra_r[0].float()/tra_r[1]).numpy(),'验证集上准确率：', (val_r[0].float()/val_r[1]).numpy())

    torch.save(cnn, 'D:/model/flower_CNN.pkl')  #保存模型
