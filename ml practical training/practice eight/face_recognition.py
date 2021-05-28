
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from torch.nn.modules.activation import ReLU, Softmax
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt

'''
重写DataLoader类
获取数据，68个人，每人49张面部照片，每张图64*64
'''
class MyDataSet(Dataset):
    def __init__(self, root, transform = None):
        super(MyDataSet, self).__init__() # 如果在子类中定义构造方法，则必须在该方法中调用父类的构造方法
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        for list in os.listdir(root): #不一定是顺序读取的
            imgpath = root + list
            img = io.imread(imgpath) # <class 'numpy.ndarray'> (64, 64)
            self.labels.append(int((int(list.split('.jpg')[0]) - 1) / 49))
            self.images.append(img)
        self.images = np.array(self.images)# <class 'numpy.ndarray'> (3332, 64, 64)
        self.images.resize(3332,1,64,64)
        self.labels = np.array(self.labels) # <class 'numpy.ndarray'> (3332,)

    def __getitem__(self, index):
        img, lb = self.images[index], self.labels[index]
        return img, lb
    def __len__(self):
        return len(self.labels)

# if __name__ == '__main__':
#     data = MyDataLoader(root = './data/FaceRecognitionDataset/')
#     img, lb = data.__getitem__(50)
#     print(type(img), img.shape, lb)
    # ax = plt.subplot(1,1,1)
    # ax.imshow(img)
    # plt.pause(60)

'''
超参数设置
'''
BATCH_SIZE = 100
EPOCH = 1000
CLASS = 68
LR = 0.0001
WEGHT_DECAY = 1e-6
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

'''
加载并划分数据集
'''
dataset = MyDataSet(root='D:/data/FaceRecognitionDataset/', transform=transforms.ToTensor())
train_size = int(len(dataset) * 0.9) # 2998
test_size = len(dataset) - train_size # 334
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE) # len = 30
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=BATCH_SIZE) # len = 4

'''
模型定义
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=4*4*64, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75, inplace=False),
            nn.Linear(in_features=1024,out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=CLASS),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        return x

def train(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEGHT_DECAY)
    for epoch in range(EPOCH):
        train_loss = 0.0
        best_loss = 100.0
        train_acc = 0.0
        for i, data in enumerate(train_loader):
            
            inputs, labs = data
            inputs = inputs.float()
            inputs = inputs.to(DEVICE)
            labs = labs.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs) # torch.tensor  BATCH_SIZE * 68
            loss = criterion(outputs, labs.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(input=outputs, dim=1)
            train_acc += sum((predicted == labs).float()) / len(labs)
            if(i + 1) % 10 == 0:
                print('Eopch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.
                format(epoch+1, EPOCH, i+1, len(train_loader), train_loss/10))
                print('Acc: {:.4f}'.format(train_acc/10))
                if(best_loss > train_loss):
                    best_loss = train_loss
                    torch.save(model.state_dict(), './trained_cnn_model.pth') # model.state_dict()只保存模型的参数， model保存整个模型
                train_loss = 0.0
                train_acc =0.0
    print('Finish training')



def test(model):
    test_acc = 0.0
    for inputs, labs in test_loader:
        inputs = inputs.float()
        inputs = inputs.to(DEVICE)
        labs = labs.to(DEVICE)
        outputs = model(inputs)
        predicted = torch.argmax(input=outputs, dim=1)
        test_acc += sum((predicted == labs).float()) / len(labs)
    print('Test Acc: {:.4f}'.format(test_acc/len(test_loader)))

if __name__ == '__main__':
    model = CNN()
    model.to(DEVICE)
    train(model)
    test(model)
    
