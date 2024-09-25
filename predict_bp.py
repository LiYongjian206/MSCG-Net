import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import numpy as np
from keras import backend as K
import time
from keras.layers import Concatenate, Add
from tensorflow.keras import optimizers, losses
import matplotlib.pyplot as plt

file = 'N'

for index in range(1, 2):

    data1 = np.load('D:/AF_BP/bp/' + file + '/test_ppg' + str(index) + '.npy', allow_pickle=True)
    data2 = np.load('D:/AF_BP/bp/' + file + '/test_ecg' + str(index) + '.npy', allow_pickle=True)
    print(data1.shape)
    print(data2.shape)
    data3 = Concatenate(axis=1)([data1, data2])

    def root_mean_square_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def mean_absolute_error(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true))

    def std(y_true, y_pred):

        return K.std(K.abs(y_true - y_pred))


    def loss(y_true, y_pred):

        return (root_mean_square_error(y_true, y_pred) + mean_absolute_error(y_true, y_pred))/2

    model = keras.models.load_model('D:/AF_BP/AF_or_BP/model/' + file + '/test_bp' + str(index) +'.hdf5',
                                    custom_objects={'loss': loss})
    start_time = time.time()
    predict1, predict2, predict3 = model.predict(data3, batch_size=256)
    # predict = model.predict(data3)
    end_time = time.time()
    run_time = end_time - start_time    # 计算运行时间（单位为秒）
    print("程序运行时间为：", run_time)

    predict = np.hstack((predict1, predict2, predict3))
    print(predict.shape)
    # pred = np.save('D:/AF_BP/bp/' + file + '/predict' + str(index) + '.npy', predict)
    plt.figure(figsize=(5, 5))
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)
    plt.plot(predict[:561, 0], label='DBP')
    plt.plot(predict[:561, 1], label='MAP')
    plt.plot(predict[:561, 2], label='SBP')
    plt.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 14})
    plt.show()

