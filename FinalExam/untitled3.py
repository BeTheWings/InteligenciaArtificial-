import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential

XTrain,XTest,YTrain,YTest = np.load('xy.npy',allow_pickle = True)
cnn=Sequential()
cnn = tf.keras.models.load_model("FinalExam.h5")
hist=cnn.fit(XTrain,YTrain,batch_size=28,epochs=100,validation_data=(XTest,YTest),verbose=2)
import matplotlib.pyplot as plt

#정확률 그래프
plt.plot(hist.history['binary_accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')

plt.ylabel('Accuracy')
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

