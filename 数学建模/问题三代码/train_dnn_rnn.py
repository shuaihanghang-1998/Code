# -*- coding: utf-8 -*-
# @Time    : 2021/10/16 10:50
# @Author  : Liu
# @File    : train_dnn.py
# @Software: PyCharm 
# @Comment :
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE
from torch.optim import Adam, lr_scheduler
from sklearn.preprocessing import normalize


ADMET_ID = 2  # ADMET特征编号  1： Caco-2  2：CYP3A4  3：hERG  4：HOB  5：MN
# img_file = 'data/Molecular_Descriptor.xlsx'
img_file = ['data/Top_50_features_1.xlsx',
            'data/Top_50_features_2.xlsx',
            'data/Top_50_features_3.xlsx',
            'data/Top_50_features_4.xlsx',
            'data/Top_50_features_5.xlsx']
label_file = 'data/ADMET.xlsx'
test_file = 'data/Molecular_Descriptor.xlsx'
N_CLASSES = 2  # 分类数目
N_FEATURES = 50
# N_FEATURES = 729
BATCH_SIZE = 32  # 每个batch要放多少张图片
EPOCH = 300
learning_rate = 0.0001
ratio = [0.8, 0.2]  # 训练集：验证集
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE: ', DEVICE)


class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        image = torch.as_tensor(image, dtype=torch.float32)
        return image, self.labels[idx]


# 划分数据集为训练集和验证集
def split_train_val_data(img_file, label_file, ratio, batch_size, resample=False):
    X_arr = pd.read_excel(img_file, sheet_name='training').iloc[:, 1:].values
    y_arr = pd.read_excel(label_file, sheet_name='training').iloc[:, ADMET_ID].values  # 1 Caco-2
    if resample:
        smo = SMOTE(random_state=1)
        X_arr, y_arr = smo.fit_resample(X_arr, y_arr)
    data = [[] for i in range(2)]  # 2分类
    for idx, row in enumerate(X_arr):
        data[y_arr[idx]].append(row)
    np.random.shuffle(data[0])
    np.random.shuffle(data[1])
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
    train_images = np.array(train_images)
    val_images = np.array(val_images)
    train_images = normalize(train_images, axis=1, norm='max')
    val_images = normalize(val_images,axis=1, norm='max')
    train_dataloader = DataLoader(MyDataset(train_images, train_labels),
                                  batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(MyDataset(val_images, val_labels),
                                batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def load_test_file (test_file, f_n):
    test_df = pd.read_excel(test_file, sheet_name='test').iloc[:, 1:]
    test_X = test_df[f_n].values
    test_X = normalize(test_X, axis=1, norm='max')
    test_X = torch.from_numpy(test_X)
    test_X = test_X.to(torch.float32)
    return test_X


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidder_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # batch_size, 1, sentence len  torch.Size([32, 1, 50])
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidder_size).to(DEVICE)  # torch.Size([2, 32, 128])
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidder_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class DNN(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super(DNN, self).__init__()
        self.l1 = torch.nn.Linear(n_features, 256)

        self.l2 = torch.nn.Linear(256, 128)
        self.d1 = torch.nn.Dropout(0.5)

        self.l3 = torch.nn.Linear(128, 64)
        self.d2 = torch.nn.Dropout(0.25)

        self.l4 = torch.nn.Linear(64, 32)
        self.d3 = torch.nn.Dropout(0.125)

        self.l5 = torch.nn.Linear(32, n_classes)

    def forward(self, x):
        x = x.view(-1, N_FEATURES)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.d1(x)
        x = F.relu(self.l3(x))
        x = self.d2(x)
        x = F.relu(self.l4(x))
        x = self.d3(x)
        return self.l5(x)


if __name__ == "__main__":
    print(torch.__version__)
    train_loader, val_loader = split_train_val_data(img_file=img_file[ADMET_ID - 1], label_file=label_file, ratio=ratio, batch_size=BATCH_SIZE, resample=True)
    model = DNN(n_classes=N_CLASSES, n_features=N_FEATURES)
    # model = RNN(input_size=N_FEATURES, hidden_size=128, num_layers=2, num_classes=N_CLASSES)
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=False,
                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        train_loss = []
        train_acc = []
        train_true_total_num = []
        loss = 0
        # 训练
        for step, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            b_x = Variable(x)
            b_y = Variable(y)
            output = model(b_x)
            loss = loss_func(output, b_y)
            pred = torch.max(output, 1)[1]
            train_T = sum(pred == y)  # 预测对的
            train_N = len(y)  # 预测总数
            train_true_total_num.append([train_T, train_N])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                print('Epoch: ', epoch, '|train loss: %.4f' % loss)
        train_r = (sum([tup[0] for tup in train_true_total_num]), sum([tup[1] for tup in train_true_total_num]))
        train_acc.append(train_r[0].float()/train_r[1])
        train_loss.append(loss)
        print('Epoch: ', epoch, '训练集上准确率：', (train_r[0].float()/train_r[1]).to('cpu').numpy())

        # 验证
        val_acc = []
        val_loss = []
        val_true_total_num = []
        for (data, target) in val_loader:
            data = data.to(DEVICE)
            # print(data.shape)  # torch.Size([32, 50]) torch.Size([10, 50])
            target = target.to(DEVICE)
            val_output = model(data)
            loss = loss_func(val_output, target)
            pred = torch.max(val_output, 1)[1]
            val_T = sum(pred == target)  # 预测对的
            val_N = len(target)  # 预测总数
            val_true_total_num.append([val_T, val_N])
            # val_r为一个二元组，分别记录验证集中分类正确的数量和该集合中总的样本数
        val_r = (sum([tup[0] for tup in val_true_total_num]), sum([tup[1] for tup in val_true_total_num]))
        val_acc.append(val_r[0].float()/val_r[1])
        val_loss.append(loss)
        print('Epoch: ', epoch, '验证集上准确率：', (val_r[0].float()/val_r[1]).to('cpu').numpy())
    # torch.save(cnn, './model/model2.pkl')  # 保存模型

    # 预测
    feature_name_20 = ['MDEC-23','C1SP2','nC','LipoaffinityIndex','minsOH','minHsOH','maxHsOH','maxsOH','SsOH',
                       'BCUTc-1l','MLogP','SHsOH','SaaCH','BCUTp-1h','MLFER_A','maxssO','BCUTc-1h','hmin','SwHBa','minsssN']

    f_n = [
        ['WPATH','ETA_Eta_R_L', 'VP-0', 'ATSm2','WTPT-1',    'SP-1',    'ECCEN',
         'ETA_Alpha',    'SP-2',    'SP-3',    'ATSm3',    'nHeavyAtom',    'MLFER_L',    'nBonds',
         'MW',    'Zagreb',    'CrippenMR',    'AMR',    'SP-0',    'VAdjMat',    'ETA_Beta_s',
         'VABC',    'ETA_Eta_F_L',    'SP-4',    'SP-5',    'ETA_Beta',    'sumI',    'apol',
         'McGowan_Volume',
         'Kier2'],
        ['SP-4',    'SP-6',    'VP-7',    'ETA_Eta_L',    'SP-3',    'VP-4',    'VP-1',
         'VP-3',    'Zagreb',    'SP-1',    'VP-0',    'ETA_Alpha',    'VP-2',    'WTPT-1',    'ETA_Eta',
         'apol',    'VAdjMat',    'ETA_Beta_s',    'SP-2',    'ETA_Eta_R_L',    'nBonds2',    'nBonds',
         'SP-0',    'ATSp3',    'SP-7',    'SP-5',    'ATSp1',    'CrippenMR',    'McGowan_Volume',
         'ETA_Eta_R'],
        ['VP-0',    'ECCEN',    'McGowan_Volume',    'bpol',    'CrippenMR',    'SP-0',
         'VP-1',    'SP-1',    'ETA_Alpha',    'VAdjMat',    'WTPT-1',    'apol',    'ETA_Eta_R_L',    'AMR',
         'Kier2',    'Kier1',    'MW',    'fragC',    'nAtom',    'ETA_Beta_s',    'VABC',    'nBondsS',
         'nH',    'ATSm2',    'minsssN',    'nBonds',    'nBonds2',    'nHeavyAtom',    'ETA_Eta_L',
         'SP-3'],
        ['nHsOH',    'nsOH',    'WTPT-5',    'VP-5',    'WTPT-4',    'SHBd',    'maxHBd',
         'BCUTp-1l',    'MDEC-22',    'WTPT-1',    'fragC',    'minaasC',    'VCH-6',    'SC-5',    'SP-2',
         'VABC',    'ETA_dEpsilon_D',    'SCH-7',    'MDEC-33',    'ETA_Epsilon_5',    'nHBAcc',    'minssO',
         'SP-5',    'maxHBint10',    'VP-4',    'SHBint10',    'ETA_Epsilon_4',    'maxdO',    'minHBint10',
         'apol'],
        [
            'WTPT-3',    'ETA_BetaP_s',    'WTPT-5',    'ETA_Epsilon_1',    'TopoPSA',    'ETA_Epsilon_4',    'nHBAcc_Lipinski',
            'ETA_dEpsilon_C',    'MDEC-24',    'ETA_dEpsilon_A',    'ETA_Epsilon_2',    'FMF',    'MLFER_S',    'ETA_EtaP_L',
            'ETA_BetaP',    'MLFER_E',    'nBondsD',    'minHBa',    'ETA_dEpsilon_B',    'ETA_EtaP_F_L',    'ETA_dPsi_A',
            'ETA_Psi_1',    'ETA_Eta_F_L',    'ETA_EtaP_F',    'XLogP',    'ETA_Epsilon_5',    'ETA_BetaP_ns',  'maxHother', 'ETA_Shape_Y',
            'hmax']
    ]
    test_X = load_test_file(test_file=test_file, f_n=f_n[ADMET_ID - 1] + feature_name_20)
    predict = []
    with torch.no_grad():
        # test_X = test_X.view(-1, N_FEATURES)
        test_X = test_X.to(DEVICE)
        outputs = model(test_X)
        _, pred = torch.max(outputs.data, 1)
        print('xxxxxxxxxxxxx', type(pred), pred.shape)
        predict.append(pred)
    print('----------------测试结果{}----------------'.format(ADMET_ID))
    print(predict)
        # print(predict[10*i : 10*(i+1)])
        # for images in test_X:
        #     X = X.reshape(-1, N_FEATURES).to(DEVICE)
        #     outputs = model(images)
        #     _, predict = torch.max(outputs.data, 1)
