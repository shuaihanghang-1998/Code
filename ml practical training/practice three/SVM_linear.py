#修改mian函数，生成第三类数据点，并将数据集划分为训练集和测试集，训练集：测试集=8:2，计算在测试集上的准确率
#测试三组数据，并输出准确率
#分别生成第一类，第二类，第三类数据点，使用SVM对其进行分类
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def mock_data(point_count=3, point_type=1):
    points = []
    labels = []
    #point_type=1，生成的数据在平面上线性可分
    if point_type == 1:
        return [[1, 3], [2, 2.5], [3.5, 1]], [0, 0, 1]
    #point_type=2，生成的数据基本线性可分，但是存在少许噪音
    elif point_type == 2:
        for i in range(point_count // 2):
            point_x = random.uniform(-5, 5)
            point_y = random.uniform(point_x+1, 5)
            points.append([point_x, point_y])
            labels.append(0)
        for i in range(point_count // 2):
            point_y = random.uniform(-5, 5)
            point_x = random.uniform(point_y+1, 5)
            points.append([point_x, point_y])
            labels.append(1)
        for i in range(point_count // 6):
            point_y = random.uniform(-5, 5)
            point_x = random.uniform(-5, 5)
            points.append([point_x, point_y])
            labels.append(random.choice([0, 1]))
    #point_type=3，生成无法线性可分的数据
    elif point_type == 3:
        for i in range(point_count // 2):
            point_x = random.uniform(-5, 5)
            point_y = random.uniform(-math.sqrt(25-point_x*point_x), math.sqrt(25-point_x*point_x))
            points.append([point_x, point_y])
            labels.append(0)
        for i in range(point_count // 2):
            point_x = random.uniform(-5, 5)
            point_y = random.choice([random.uniform(-25, -math.sqrt(25-point_x*point_x)), random.uniform(math.sqrt(25-point_x*point_x), 25)])
            points.append([point_x, point_y])
            labels.append(1)
    return points, labels
#绘制分类结果及超平面
def plot_point(dataArr, labelArr, Support_vector_index, W=0, b=0):
    for i in range(np.shape(dataArr)[0]):
        if labelArr[i] == 1:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='b', s=20)
        else:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='y', s=20)
    for j in Support_vector_index:
        plt.scatter(dataArr[j][0], dataArr[j][1], s=100, c='',edgecolors='r',  marker='o', alpha=0.5, linewidths=1)
    x = np.arange(-5, 5, 0.01)
    y = (W[0][0] * x + b)/(-1 * W[0][1])
    plt.scatter(x, y, s=5, marker='h')
    plt.show()
if __name__ == '__main__':
    #分别生成第一类，第二类，第三类数据
    #补充代码
    dataArr, labelArr=mock_data(point_count=100, point_type=3)
    x_train, x_test, y_train, y_test = train_test_split(dataArr, labelArr, train_size=0.8, test_size=0.2, random_state=1)
    #定义SVM分类器，核函数定义为线性核函数，其他参数使用默认值
    #补充代码
    clf=SVC(kernel='linear',C=1,decision_function_shape='ovr')
    #fit训练数据
    clf.fit(x_train, y_train)
    #获取模型返回值
    n_Support_vector = clf.n_support_   #支持向量个数
    Support_vector_index = clf.support_ #支持向量索引
    W = clf.coef_   #方向向量
    b = clf.intercept_  #截距项b
    #绘制分类超平面
    plot_point(x_train, y_train, Support_vector_index, W, b)
    y_test_hat = clf.predict(x_test)
    result = (y_test_hat == y_test)
    score = np.mean(result)
    print( u'accuracy_score: %.2f%%' % (100 * score))