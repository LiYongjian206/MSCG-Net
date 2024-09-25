import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



data = np.load('D:/AF_BP/af/af/AFDB/data.npy', allow_pickle=True)
label = np.load('D:/AF_BP/af/af/AFDB/label.npy', allow_pickle=True)
label = np_utils.to_categorical(label, 2)
# 1 16
# 2 8
# 3 32
# 4 64
# 5 128
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=128)

model = keras.models.load_model('D:/AF_BP/AF_or_BP/model/AFDB5.hdf5')
acc = model.evaluate(X_test, y_test, batch_size=512, verbose=1)

F1 = []
Con_Matr = []

y_pred = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
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