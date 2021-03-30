import numpy as np
import random
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
np.random.seed()
percentage=0.2
iter=10

def prepare_data():
    iris = datasets.load_iris()
    x = iris.data[:, [0, 1]]
    y = iris.target
    return x,y

#kmeans聚类
def eva_kmeans(x,y):
    #运行十次，取平均分类正确样本数，运行时间，准确率
    kmean_ei = 0.0  #分类正确样本数
    kmean_rt = 0.0  #运行时间
    kmean_aa = 0.0  #准确率
    for i in range(0, iter):
        x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                            train_size=0.8, 
                                                            test_size=0.2, 
                                                            random_state=1)
        k_begin = time.time()
        kmeans = KMeans(init="random", n_clusters=3, random_state=0).fit(x_train)       
        kmean_pred = kmeans.predict(x_test)
        k_end = time.time() - k_begin
        kmean_rt = kmean_rt + k_end
        accuracy_number = 0
        for i in range(len(y_test)):
            if kmean_pred[i]==1:
                kmean_pred[i]=0
            else:
                if kmean_pred[i]==2: 
                    kmean_pred[i]=1
                else:
                    if kmean_pred[i]==0: 
                        kmean_pred[i]=2
                        '''
                        调整分类
                        '''
            if kmean_pred[i] == y_test[i]:
                accuracy_number += 1
        kmean_ei = kmean_ei + accuracy_number
        accuracy_percentage = metrics.accuracy_score(y_test, kmean_pred)*100      
        kmean_aa = kmean_aa + accuracy_percentage
    kmean_ei = kmean_ei / (iter*1.0)
    kmean_rt = kmean_rt / (iter*1.0)
    kmean_aa = kmean_aa / (iter*1.0)
    for i in range(len(kmean_pred)):
        if kmean_pred[i] == 0:
            plt.scatter(x_test[i][0], x_test[i][1], c='b', s=20)
        elif kmean_pred[i] == 1:
            plt.scatter(x_test[i][0], x_test[i][1], c='y', s=20)
        else:
            plt.scatter(x_test[i][0], x_test[i][1], c='g', s=20)
    plt.show()
    for i in range(len(y_test)):
        if y_test[i] == 0:
            plt.scatter(x_test[i][0], x_test[i][1], c='b', s=20)
        elif y_test[i] == 1:
            plt.scatter(x_test[i][0], x_test[i][1], c='y', s=20)
        else:
            plt.scatter(x_test[i][0], x_test[i][1], c='g', s=20)
    plt.show()
    return kmean_ei, kmean_rt, kmean_aa

#层次聚类
def eva_hierarchical(x,y):
    hier_ei = 0.0
    hier_rt = 0.0
    hier_aa = 0.0
    for i in range(0, 10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=1)
        k_begin = time.time()
        hier = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="average").fit(x_train) 
        hier_pred = hier.fit_predict(x_test)
        k_end = time.time() - k_begin
        hier_rt = hier_rt + k_end
        accuracy_number = 0
        for i in range(len(y_test)):
            if hier_pred[i]==1:
                hier_pred[i]=2
            else:
                if hier_pred[i]==2: 
                    hier_pred[i]=0
                else:
                    if hier_pred[i]==0: 
                        hier_pred[i]=1
                        '''
                        调整分类
                        '''
            if hier_pred[i] == y_test[i]:
                accuracy_number += 1
        hier_ei = hier_ei + accuracy_number
        accuracy_percentage = metrics.accuracy_score(y_test, hier_pred)*100
        hier_aa = hier_aa + accuracy_percentage
    hier_ei = hier_ei / (iter*1.0)
    hier_rt = hier_rt / (iter*1.0)
    hier_aa = hier_aa / (iter*1.0)
    for i in range(len(hier_pred)):
        if hier_pred[i] == 0:
            plt.scatter(x_test[i][0], x_test[i][1], c='b', s=20)
        elif hier_pred[i] == 1:
            plt.scatter(x_test[i][0], x_test[i][1], c='y', s=20)
        else:
            plt.scatter(x_test[i][0], x_test[i][1], c='g', s=20)
    plt.show()
    return hier_ei, hier_rt, hier_aa

if __name__ == '__main__':
    x,y=prepare_data()
    kmean_ei = np.zeros(100) 
    kmean_rt = np.zeros(100) 
    kmean_aa = np.zeros(100) 
    '''
    for i in range(100):
        print(i)
        kmean_ei[i], kmean_rt[i], kmean_aa[i]=eva_kmeans(x,y,i)
    '''
    kmean_ei, kmean_rt, kmean_aa=eva_kmeans(x,y)
    
    hier_ei, hier_rt, hier_aa = eva_hierarchical(x, y)
    print ("total iterate:",iter)
    print ("method   ",  "# of error instances     ",  "run_time/s           ", "accuracy/%")
    '''
    for j in range(100):
        print ("K-means          ",  kmean_ei[j],"          ",  kmean_rt[j],"      ",  kmean_aa[j])
    '''
    print ("K-means          ",  kmean_ei,"          ",  kmean_rt,"      ",  kmean_aa)

    print ("hierarchical      ", hier_ei, "          ", hier_rt, "       ", hier_aa)