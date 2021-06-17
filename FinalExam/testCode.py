import os, re, glob
import cv2
import numpy as np
import shutil
from numpy import argmax
from tensorflow.keras.models import load_model

path= './test/'
Categories = os.listdir(path)
Categories
categories=["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]

def Dataization(Categories):
    image_w = 28
    image_h = 28
    img = cv2.imread(Categories)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)

print(1) 
src = []
name = []
test = []

for file in Categories:
    if (file.find('.png') is not -1):      
        src.append(path + file)
        name.append(file)
        test.append(Dataization(path + file))
       
       
test = np.array(test)
model = load_model('FinalExam.h5')
predict = model.predict_classes(test)


for i in range(len(test)):
    print(name[i] + " : , Predict : "+ str(categories[predict[i]]))

