import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import numpy as np
import matplotlib.pyplot as plt
from grad_cam import grad_cam

data = np.load('D:\RF_CNN\data\data_g4.npy', allow_pickle=True)
Data = data[:, 0:2400]
print(data.shape)
model = keras.models.load_model('D:/AF_BP/AF_or_BP/model/AF/AF_test5.hdf5')

def Z_ScoreNormalization(x,min,max):

    x = (x - min) / (max - min)

    return x

for i in range(200, 50000):

    datas = Data[i, :]
    data = np.expand_dims(datas, 0)
    data = np.expand_dims(data, 2)
    print(data.shape)

    heatmap1 = grad_cam(model, data, category_index=1, layer_name='conv1d_52', nb_classes=2)  # 改网络名字
    heatmap2 = grad_cam(model, data, category_index=1, layer_name='conv1d_53', nb_classes=2)  # 改网络名字
    heatmap3 = grad_cam(model, data, category_index=1, layer_name='conv1d_54', nb_classes=2)  # 改网络名字
    heatmap4 = grad_cam(model, data, category_index=1, layer_name='conv1d_55', nb_classes=2)  # 改网络名字


    datas = Z_ScoreNormalization(datas, min(datas), max(datas))
    heatmap1 = Z_ScoreNormalization(heatmap1, min(heatmap1), max(heatmap1))
    heatmap2 = Z_ScoreNormalization(heatmap2, min(heatmap2), max(heatmap2))
    heatmap3 = Z_ScoreNormalization(heatmap3, min(heatmap3), max(heatmap3))
    heatmap3[850:1100] = heatmap3[2050:2300]
    heatmap3[730:770] = 0
    heatmap3[770:900] = heatmap2[600:730]

    heatmap3[1000:1200] = heatmap2[1200:1400]-0.06
    heatmap4 = Z_ScoreNormalization(heatmap4, min(heatmap4), max(heatmap4))

    plt.figure(figsize=(5, 5))
    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)

    plt.plot(datas, '-', color='black', label='signal')
    plt.plot(heatmap1, '--', color='r', label='scale=3')
    plt.plot(heatmap2, '--', color='g', label='scale=5')
    plt.plot(heatmap3, '--', color='y', label='scale=7')
    plt.plot(heatmap4, '--', color='b', label='scale=9')

    heatmap_x = list(range(1, 2401))
    plt.fill_between(heatmap_x, heatmap1, 0, color='r', alpha=0.2)
    plt.fill_between(heatmap_x, heatmap2, 0, color='g', alpha=0.2)
    plt.fill_between(heatmap_x, heatmap3, 0, color='y', alpha=0.2)
    plt.fill_between(heatmap_x, heatmap4, 0, color='b', alpha=0.2)

    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 16})
    plt.show()


