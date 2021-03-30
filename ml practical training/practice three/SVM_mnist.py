import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pdb
import math
import random
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml


#digits = datasets.load_digits()
#n_samples = len(digits.images)
#data = digits.images.reshape((n_samples, -1))


mnist = fetch_openml('mnist_784')
x = mnist['data']
y = mnist['target']

# Create a classifier: a support vector classifier
clf = svm.SVC(C=1,decision_function_shape='ovo')

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False)
X_train, X_test=X_train/255.0, X_test/255.0

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1, ncols=8, figsize=(10, 3))
for ax in axes:
    ax.set_axis_off()
    x_show=np.array(X_test)
    predicted=np.array(predicted)
    i= random.uniform(0,13999)
    image=x_show[int(i),:]
    image= image.reshape(28, 28)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'prediction: {predicted[int(i)]}')
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")

'''
#plot a confusion matrix of the true digit values and the predicted digit values.
disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
'''
plt.show()
