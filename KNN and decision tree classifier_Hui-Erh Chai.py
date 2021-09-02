# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 09:03:10 2021

@author: lenovo
"""
# import datasets
import pandas as pd
df = pd.read_csv('Treasury Squeeze raw score data.csv', sep = ',')

# mapping
#df = df.map({'True':0, 'False':1})

# Identify X and y
X = df.iloc[:, 2:-1].values
y = df.iloc[:, -1].values

# mapping
#y = y.map({'True':0, 'False':1})


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=1, stratify=y)


# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'greenyellow']
for i in range(len(colors)):
    Xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(Xs, ys, c=colors[i])
#plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# try k=1 through k=25 and record testing accuracy
k_range = range(1, 26)
scores = []
error_rate = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
 
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
#import numpy as np
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,pred))



print("My name is Hui-Erh Chai_Angela")
print("My NetID is: 674939884")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
