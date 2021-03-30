import numpy as np
import matplotlib.pyplot as plt

def k_means(x, k=3):
    index_list = np.arange(len(x))
    np.random.shuffle(index_list)
    centroids_index = index_list[:k]
    centroids = x[centroids_index]
    y = np.arange(len(x))
    iter_num = 10000 #自行设置迭代次数
    for i in range(iter_num):
        y_new = np.arange(len(x))
        for i, xi in enumerate(x):
            '''
            补充代码，计算x中每个点属于哪个簇心，
            计算距离用np.linalg.norm()，计算簇心用np.argmin()
            '''
            y_new[i] = np.argmin([np.linalg.norm(xi - cj) 
                                 for cj in centroids])
        if sum(y != y_new) == 0:
            break
        for j in range(k):
            '''
            补充代码，重新计算簇心，使用np.mean
            '''
            centroids[j] = np.mean(x[np.where(y_new == j)], axis=0)
        y = y_new.copy()
    return y

if __name__ == '__main__':
    #可自行创造其他数据
    x = np.array([[1,1],[3,3],[1,7],[2,8],[10,20],[1,30],[2,29],[10,11],[15,18],[12,19]])
    y = k_means(x)
    print(y)
    #可视化
    for i in range(len(y)):
        if y[i] == 0:
            plt.scatter(x[i][0], x[i][1], c='b', s=20)
        elif y[i] == 1:
            plt.scatter(x[i][0], x[i][1], c='y', s=20)
        else:
            plt.scatter(x[i][0], x[i][1], c='g', s=20)
    plt.show()
