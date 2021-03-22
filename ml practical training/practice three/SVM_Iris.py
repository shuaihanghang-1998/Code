from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# Training classifiers
clf = SVC(gamma=.1, kernel='rbf', probability=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
clf.fit(x_train, y_train)


# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
plt.title("Iris")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # SVM的分割超平面

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train)

sv = clf.support_vectors_
plt.scatter(sv[:, 0], 
            sv[:, 1],
            s=100, 
            c='',
            edgecolors='r',  
            marker='o',
            alpha=0.5, 
            linewidths=1)

y_test = y_test.reshape(-1)
y_test_hat = clf.predict(x_test)
result = (y_test_hat == y_test)
score = np.mean(result)
print('accuracy_score:', score)

plt.show()