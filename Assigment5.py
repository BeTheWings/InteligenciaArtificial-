# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 02:47:11 2021

@author: JeesooPark
"""
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

lineA=[]
svmA=[]
treeA=[]
digit = datasets.load_digits()
s=svm.SVC(gamma=0.001)
for i in range(6):
    SVMaccuracies = cross_val_score(s, digit.data, digit.target,cv=(5+i))
    print(SVMaccuracies)
    print("SVM정확률(평균)=%0.3f,표준편차=%0.3f"%(SVMaccuracies.mean()*100,SVMaccuracies.std())) 
    svmA.append(SVMaccuracies.mean()*100)    
    lineA.append(5+i)
print(svmA)
X, y = datasets.load_digits(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
"""
tree.plot_tree(clf,max_depth=1)
"""

for i in range(6):
    accuracies = cross_val_score(clf, X, y,cv=(5+i))
    print(accuracies)
    print("Decision정확률(평균)=%0.3f,표준편차=%0.3f"%(accuracies.mean()*100,accuracies.std()))
    treeA.append(accuracies.mean()*100)    

plt.plot(lineA,treeA,lineA,svmA,marker='o')
plt.xlabel('num of cv')
plt.ylabel('mean of acc')
plt.title('compare acc')
plt.legend(['tree', 'svm'])
plt.show()


