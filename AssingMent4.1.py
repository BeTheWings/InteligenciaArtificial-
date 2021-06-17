# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 21:04:35 2021

@author: JeesooPark
"""
from sklearn import svm, datasets    
import pandas  as pd
import random
from sklearn.metrics import accuracy_score
d = datasets.load_iris()
data = d.data
target = d.target
RandomData = pd.DataFrame(data).sample(20)
num = target[RandomData.index]
RandomDataTest = RandomData.sample(frac=0.05)
RandomDataTest
test = RandomData.copy()
test.loc[RandomDataTest.index] = [[6.4,3.2,6.0,2.5]]
s = svm.SVC(gamma=0.1,C=10)
s.fit(RandomData,num)
res = s.predict(test)
print("새로운 1개의 샘플의 부류는",res)
print("정확도는",accuracy_score(num,res),"입니다.")
