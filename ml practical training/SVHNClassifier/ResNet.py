### code
import os
from PIL import Image
from torch.utils import data
import numpy as np
import warnings
from torchvision import transforms
import csv
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import pandas as pd
from torch.utils.data import Dataset


class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):

        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        lbl = np.array(self.img_label[index])
        return img, torch.from_numpy(lbl).squeeze().long()
        
    def __len__(self):
        return len(self.img_path)

class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = torchvision.models.resnet18(pretrained=False)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 10)


    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)

        return c1

def get_path_label(img_root, label_file_path):
    """
    获取数字图像的路径和标签并返回对应列表
    @para: img_root: 保存图像的根目录
    @para:label_file_path: 保存图像标签数据的文件路径 .csv 或 .txt 分隔符为','
    @return: 图像的路径列表和对应标签列表
    """
    imgs = [os.path.join(img_root, img) for img in os.listdir(img_root)]
    imgs.sort(key=lambda x : x.split('.jpg')[0])
    label = pd.read_csv(label_file_path)
    return imgs, label.values.tolist()
def train(train_loader, model, criterion, optimizer):
    # 切换模型为训练模式
    model.train()
    train_loss = []
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        c0= model(input)
        loss = criterion(c0, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    return np.mean(train_loss)

def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            c0 = model(input)
            loss = criterion(c0, target)
            val_loss.append(loss.item())
    return np.mean(val_loss)

def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None

    # TTA 次数
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()

                c0 = model(input)
                if use_cuda:
                    output = np.concatenate([c0.data.cpu().numpy(),], axis=1)
                else:
                    output = np.concatenate([c0.data.numpy()], axis=1)

                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta



# 获取训练集路径列表和标签列表
train_data_root = 'D:\\Code\\ml practical training\\SVHNClassifier\\data\\train'
train_label = 'D:\\Code\\ml practical training\\SVHNClassifier\\data\\train_labels.csv'
train_img_list, train_label_list = get_path_label(train_data_root, train_label)  


# 获取测试集路径列表和标签列表
test_data_root = 'D:\\Code\\ml practical training\\SVHNClassifier\\data\\test'
test_label = 'D:\\Code\\ml practical training\\SVHNClassifier\\data\\test_labels.csv'
test_img_list, test_label_list = get_path_label(test_data_root, test_label)

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_img_list, train_label_list,
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])),
    batch_size=50,
    shuffle=True,
    num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        SVHNDataset(test_img_list, test_label_list,
                    transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])),
        batch_size=50,
        shuffle=False,
        num_workers=4,
    )
    model = SVHN_Model1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.1)

    use_cuda = False
    if use_cuda:
        model = model.cuda()

    best_loss = 1000.0
    for epoch in range(20):
        train_loss = train(train_loader, model, criterion, optimizer)
        val_loss = validate(val_loader, model, criterion)
        val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        val_predict_label = predict(val_loader, model, 1)
        val_predict_label = np.vstack([val_predict_label[:, :10].argmax(1)]).T
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x)))

        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

        print('Epoch: {0}, Train loss: {1} -- Val loss: {2} -- Val Acc: {3}'.format(epoch, train_loss, val_loss, val_char_acc))

        # 记录下验证集精度
        if val_loss < best_loss:
            best_loss = val_loss
            print('Find better model in Epoch {0}, saving model.'.format(epoch))
            torch.save(model.state_dict(), './model.pt')