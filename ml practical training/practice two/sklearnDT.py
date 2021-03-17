from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from numpy import *
import matplotlib.pyplot as plt

data = datasets.load_iris()
x = data.data
y = data.target
x_2 = x[:, :2] # only 2 features
x_4 = x[:, :4] # 4 features

x_train2, x_test2, y_train2, y_test2 = train_test_split(x_2, y, train_size=0.7, test_size=0.3, random_state=1)
'''
补充代码，参考上句将x_4分为训练集和测试集
'''
x_train4, x_test4, y_train4, y_test4 = train_test_split(x_4, y, train_size=0.7, test_size=0.3, random_state=1)
Accuracy2 = []
Accuracy4 = []
for i in range(15):
    '''
    补充代码，调用sklearn决策树类，传入参数depth；
    对x_2训练集进行训练；
    输出测试集结果；
    计算准确度。
    '''
    depth = i+1
    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    dtc.fit(x_train2, y_train2)
    score2 = dtc.score(x_test2, y_test2)
    print("Depth: " , depth," | Accuracy:",score2)
    Accuracy2.append(score2)
for i in range(15):
    '''
    补充代码，调用sklearn决策树类，传入参数depth；
    对x_4训练集进行训练；
    输出测试集结果；
    计算准确度。
    '''
    depth = i+1
    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    dtc.fit(x_train4, y_train4)
    score4 = dtc.score(x_test4, y_test4)
    print("Depth: " , depth," | Accuracy:",score4)
    Accuracy4.append(score4)

t = list(range(1,16))
plt.figure()
plt.plot(t,Accuracy2)
plt.plot(t,Accuracy4)
plt.xlabel('depth')
plt.ylabel('Accuracy')
plt.ylim(0.5,1)
plt.legend(['2-features','4-features'])
plt.show()