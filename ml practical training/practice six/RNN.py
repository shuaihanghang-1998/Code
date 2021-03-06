import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pdb
import torch
from torch import nn
from torch.autograd import Variable


dataset=[]
for data in np.arange(0, 3, .01):
    data = math.sin(data*math.pi)
    dataset.append(data)
    
dataset=np.array(dataset)
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))

def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

data_X, data_Y = create_dataset(dataset)

train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]



train_X = train_X.reshape(-1, 1, 3)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 3)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
# 定义模型

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=5):
        super(RNN, self).__init__()
        
        #self.rnn = nn.LSTM(input_size, hidden_size, num_layers) # LSTM
        self.rnn = nn.RNN(input_size, hidden_size, num_layers) # RNN
        #self.rnn = nn.GRU(input_size, hidden_size, num_layers) # GRU
        self.linear = nn.Linear(hidden_size, output_size) # 回归
        
    def forward(self, x):
        #pdb.set_trace()
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.linear(x)
        x = x.view(s, b, -1)
        return x

net = RNN(3, 20)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
# 开始训练
for e in range(500):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0: # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.10f}'.format(e + 1, loss.data.item()))
net = net.eval() # 转换成测试模式
data_X = data_X.reshape(-1, 1, 3)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = net(var_data) # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()
# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset[2:], 'b', label='real')
plt.legend(loc='best')
plt.savefig("test1.png")
plt.show()