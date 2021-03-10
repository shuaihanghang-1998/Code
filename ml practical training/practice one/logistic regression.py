#利用逻辑回归对西瓜分类
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
#sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升函数
def gradAscent(dataMat, labelMat):
    m,n = shape(dataMat)
    print(m, n)
    alpha = 0.1
    maxCycles = 500
    weights = array(ones((n, 1)))
    for k in range(maxCycles):
        for i in range(m):
            h=weights[0]+weights[1]*dataMat[i,1]+weights[2]*dataMat[i,2]
            err=labelMat[i]-sigmoid(h)
            for j in range(n):
                weights[j] = weights[j] +alpha * err * dataMat[i,j]
    return weights

#随机梯度上升
def randomgradAscent(dataMat, label, numIter=50):
    m,n = shape(dataMat)
    weights = ones(n)
    for k in range(numIter):
       #dataIndex = list(range(m))
        for i in range(m):
            #迭代次数越多，学习率越小
            alpha = 40/(1.0+k+i)+0.2
            index=random.uniform(0,m)
            dataindex=int(index)
            h=weights[0]+weights[1]*dataMat[dataindex,1]+weights[2]*dataMat[dataindex,2]
            err=labelMat[dataindex]-sigmoid(h)
            for j in range(n):
                weights[j] = weights[j] +alpha * err * dataMat[dataindex,j]
    return weights

#画图
def plotBestFit(weights1,weights2,weights3):
    m = shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i, 1])
            ycord1.append(dataMat[i, 2])
        else:
            xcord2.append(dataMat[i, 1])
            ycord2.append(dataMat[i, 2])
    plt.figure(1)
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, c='green')
    x = arange(0.2, 0.8, 0.1)
    y1 = array((-weights1[0] - weights1[1] * x)/weights1[2])
    y2 = array((-weights2[0] - weights2[1] * x)/weights2[2])    
    y3 = array((-weights3[0] - weights3[1] * x)/weights3[2])
    
    plt.sca(ax)
    #plt.plot(x, y[0])   #gradAscent
    plt.plot(x, y1, 'r-',label=u'numIter=50')  #randmgradAscent
    plt.plot(x, y2, 'b-', label=u'numIter=20')
    plt.plot(x, y3, 'g-', label=u'numIter=10')
    plt.legend(loc='upper right')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    #plt.title('gradAscent logistic regression')
    plt.title('random gradAscent logistic regression')
    plt.show()

#读入csv文件数据
df = pd.read_csv('/data/shixunfiles/03d07802327d2fae5cbde9b1466fe53e_1582714323718.csv')
m, n = shape(df)
#df的第一列为index，在实验原理中该列的值应当为1，所以采用“idx”索引df对其修改
df['idx'] = ones((m, 1))
dataMat = array(df[['idx', 'idensity', 'ratio_sugar']].values[:, :])
labelMat = mat(df['label'].values[:]).transpose()
#weights = gradAscent(dataMat, labelMat)
weights1 = randomgradAscent(dataMat, labelMat, numIter=50)
weights2 = randomgradAscent(dataMat, labelMat, numIter=20)
weights3 = randomgradAscent(dataMat, labelMat, numIter=10)
plotBestFit(weights1,weights2,weights3)