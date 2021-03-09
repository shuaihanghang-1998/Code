#岭回归预测销售额
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

#岭回归预测销售额
data = pd.read_csv('/data/shixunfiles/f4a77266e4adb25e4ab168f994f6a7e6_1582712806305.csv')
x = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
#划分数据集，80%用于训练数据集，20%用于测试数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,test_size=0.2, random_state=0)

#sklearn库中岭回归模型
r = Ridge()
ridge = r.fit(x_train, y_train)
print('Training set score:{}'.format(ridge.score(x, y)))
print('riddge.coef_:{}'.format(ridge.coef_))
print('ridge.intercept_:{}'.format(ridge.intercept_))
order = y_test.argsort(axis=0)
y_test = y_test.values[order]
x_test = x_test.values[order, :]
y_predict = r.predict(x_test)
#计算测试集上的均方误差
mse = np.average((y_predict - np.array(y_test)) ** 2)
#计算测试集上的均方根误差
rmse = np.sqrt(mse)
print('MSE= ', mse)
print('RMSE= ', rmse)
#绘制岭回归销售额和真实销售额折线图
plt.figure(facecolor='w')
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label=u'real data')
plt.plot(t, y_predict, 'b-', linewidth=2, label=u'predicted data')
plt.legend(loc='upper right')
plt.title(u'predict sales by ridge regression', fontsize=18)
plt.grid(b=True)
plt.show()
#自己编写岭回归算法
x_T = np.transpose(x_train)
num = 20000
k_mat = np.linspace(-10000, num-1-10000, num=num)
beta = np.zeros([num, 3])
i=0
beta = np.linalg.pinv(x_train.T.dot( x_train) + (5000)*np.eye(3)) .dot(x_train.T).dot(y_train)
#for lamda in k_mat:
    #beta[i] = np.linalg.pinv(x_train.T.dot( x_train) + lamda*np.eye(3)) .dot(x_train.T).dot(y_train)
    #i=i+1
print('beta= ',beta)
plt.plot(beta)
plt.show()
#a = [0.05263455, 0.25295774, 0.00258252]
y = np.dot(x_test, beta)
plt.plot(y_test, 'r-', linewidth=2, label=u'real data')
plt.plot(y_predict, 'b-', linewidth=2, label=u'predicted data')
plt.plot(y,'g-',linewidth=2, label=u'my predicted data')
plt.legend(loc='upper right')
plt.title(u'my prediction', fontsize=18)
plt.show()