# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix  # 混淆矩阵
import time


data = np.load('D:\RF_CNN\data\data_g4.npy', allow_pickle=True)
label = np.load('D:\RF_CNN\data\label_g4.npy', allow_pickle=True)
data = data[:, 0:2400]
print(data.shape)
label = np_utils.to_categorical(label, 2)

model = keras.models.load_model('D:/AF_BP/AF_or_BP/model/AF/AF_test5.hdf5')
start_time = time.time()
acc = model.evaluate(data, label, batch_size=512, verbose=1)
end_time = time.time()
run_time = end_time - start_time
print(run_time)

F1 = []
Con_Matr = []

y_pred = model.predict(data)
y_test = np.argmax(label, axis=1)
y_pred = np.argmax(y_pred, axis=1)
con_matr = confusion_matrix(y_test, y_pred)
Con_Matr.append(con_matr)
Con_Matr = np.array(con_matr)
sen = Con_Matr[1, 1] / (Con_Matr[1, 1] + Con_Matr[1, 0])
spe = Con_Matr[0, 0] / (Con_Matr[0, 0] + Con_Matr[0, 1])
pre = Con_Matr[1, 1] / (Con_Matr[1, 1] + Con_Matr[0, 1])
f1 = 2*sen*pre/(sen+pre)

print(spe)
print(sen)
print(pre)
print(f1)