#将SVM的参数中的核函数修改为rbf，测试若干组数据，观察并分析准确率
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
            point_x = random.uniform(0, 10)
            point_y = random.uniform(point_x+1, 10)
            points.append([point_x, point_y])
            labels.append(0)
        for i in range(point_count // 2):
            point_y = random.uniform(0, 10)
            point_x = random.uniform(point_y+1, 10)
            points.append([point_x, point_y])
            labels.append(1)
        for i in range(point_count // 6):
            point_y = random.uniform(0, 10)
            point_x = random.uniform(0, 10)
            points.append([point_x, point_y])
            labels.append(random.choice([0, 1]))
    #point_type=3，生成无法线性可分的数据
    elif point_type == 3:
        for i in range(point_count // 2):
            point_x = random.uniform(-2, 2)
            point_y = random.uniform(-math.sqrt(4-point_x*point_x), math.sqrt(4-point_x*point_x))
            points.append([point_x, point_y])
            labels.append(0)
        for i in range(point_count // 2):
            point_x = random.uniform(-2, 2)
            point_y = random.choice([random.uniform(-4, -math.sqrt(4-point_x*point_x)), random.uniform(math.sqrt(4-point_x*point_x), 4)])
            points.append([point_x, point_y])
            labels.append(1)
    return points, labels
#绘制分类结果及超平面
def plot_hyperplane(clf, X, y,
                    h=0.02,
                    draw_sv=True,
                    title='hyperplan'):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # SVM的分割超平面

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)
    plt.scatter(X[:,0],X[:,1],c=y)


    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1],s=100, c='',edgecolors='r',  marker='o', alpha=0.5, linewidths=1)
        #plt.scatter(dataArr[j][0], dataArr[j][1], s=100, c='',edgecolors='r',  marker='o', alpha=0.5, linewidths=1)

    plt.show()
if __name__ == '__main__':
    #分别生成第一类，第二类，第三类数据
    #补充代码
    dataArr, labelArr = mock_data(point_count=100, point_type=1)
    x_train, x_test, y_train, y_test = train_test_split(dataArr, labelArr, train_size=0.8, test_size=0.2, random_state=1)
    #定义SVM分类器
    #补充代码
    clf = SVC(kernel='rbf',C=1,decision_function_shape='ovr')
    #fit训练数据
    clf.fit(x_train, y_train)
    point_type = 2
    kernel = 'rbf'
    x_arr_train = np.array(x_train)
    y_arr_train = np.array(y_train)
    plot_hyperplane(clf, x_arr_train, y_arr_train, h=0.01, title="point_type={}  ,  kernel={}".format(2, kernel))
    y_test_hat = clf.predict(x_test)
    result = (y_test_hat == y_test)
    score = np.mean(result)
    print('accuracy_score:', score)