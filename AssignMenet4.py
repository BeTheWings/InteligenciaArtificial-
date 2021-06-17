# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:29:13 2021

@author: 정 민
"""
# 붓꽃의 종류인 target은 0, 1, 2로 되어 있는데 이는 각각 setosa, versicolor, virginica를 나타낸다.

from sklearn.datasets import load_iris
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

d = load_iris()
iris_data = d.data
iris_label = d.target
iris_label_name = d.target_names

new_data = pd.DataFrame(iris_data).sample(20) #랜덤 20개 추출
target = iris_label[new_data.index] #20개 뽑은 것의 target

print(target)

data_test = new_data.sample(frac = 0.05) #5%이내 변형하기 위한 추출
print("----------------------")
print(data_test)

data = new_data.copy()
print(data)

data.loc[data_test.index] = [[3.5, 3.7, 4.2, 1.8]] #데이터 변형

s = svm.SVC(gamma = 0.1, C = 10)
s.fit(new_data, target)

res = s.predict(data) # 변형한 데이터 예측
print("샘플 1개의 부류 : ", res)
print("정확도는",accuracy_score(target,res),"입니다.")

