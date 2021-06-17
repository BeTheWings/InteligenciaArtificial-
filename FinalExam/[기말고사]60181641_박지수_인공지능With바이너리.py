# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 02:46:37 2021

@author: JeesooPark
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:22:34 2021

@author: JeesooPark
"""
import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL  import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
from tensorflow.keras import metrics

#이미지의 크기를 조절하기 위한 변수
ImageW=28
ImageH= 28
X=[]
Y=[]
#현재 이미지가 위치한 Location
path= './data/'
#os를 통해 경로상 위치한 모든 폴더를 Categories에 저장합니다.
Categories = os.listdir(path)
#폴더의 총 수를 Length에 저장합니다.
Length = len(Categories)

#반복문을 통해 Categories안에 있는 내용들을 정리합니다.
for idx, categorie in enumerate(Categories):
    #원핫 코딩으로 True외의 값을 0으로 변경합니다.
    label = [0 for i in range(Length)]
    #원핫 코딩으로 True값은 1이 됩니다.
    label[idx] = 1
    
    image_dir_detail = path + "/" + categorie
    #경로에서 png파일을 읽어옵니다.
    files = glob(image_dir_detail + "/*.png")
    #총 사진파일의 수를 출력합니다.
    print(categorie, "파일 길이 : ", len(files))
    
    for i, f in enumerate(files):
        #이미지를 열어서 RGB로 Convert해준 후 resize를 통해 이미지의 크기를 28*28로 바꿔줍니다.
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((ImageW, ImageH))
        data = np.asarray(img)

        X.append(data)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)
#X,Y에 저장된 데이터를 4개의 DATA로 나눈후 xy변수에 저장하여 npy파일 형태로 저장합니다.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
xy = (X_train, X_test, Y_train, Y_test)
 
np.save('xy.npy', xy)



#파일의 위치를 선정하여 train data와 test데이터를 만들어 줍니다.
#pickle오류를 해결하기 위해 True를 사용하였습니다.
XTrain,XTest,YTrain,YTest = np.load('xy.npy',allow_pickle = True)
#용량을 제어하기 위해 이미지의 크기를 28 * 28로 줄여줍니다.#
XTrain=XTrain.astype(np.float32)/255.0
XTest=XTest.astype(np.float32)/255.0



#신경망 모델 설정
cnn=Sequential()
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,3)))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(Length,activation='softmax'))
#categorical crossentory보다 binary corssentropy의 성능이 더 좋아서 binary를 적용하였습니다.
#binary corss enrtopy란?
# 바이너리 크로스엔트로피를 사용한 이유는 현재 
#내가 학습을 진행하면서 제공한 사진이 맞는지 아닌지 파악하는 이진 문제이기 때문에 binary cross entropy가 더 적합하다고 생각하여 다음과 같이 적용하게 되었습니다.
cnn.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
hist=cnn.fit(XTrain,YTrain,batch_size=28,epochs=100,validation_data=(XTest,YTest),verbose=2)

res = cnn.evaluate(XTest,YTest,verbose=0)
print("정확률은",res[1]*100)
cnn.save("FinalExam.h5")
import matplotlib.pyplot as plt

#정확률 그래프
plt.plot(hist.history['binary_accuracy'])
plt.plot(hist.history['val_binary_accuracy'])
plt.title('Model accuracy')

plt.ylabel('binary_accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

#손실 함수 그래프
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()
