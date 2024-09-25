import numpy as np
from keras import Input
from keras.callbacks import LearningRateScheduler
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Add, Multiply, \
    GlobalAveragePooling1D, Concatenate, Reshape, BatchNormalization, ELU, DepthwiseConv1D, AveragePooling1D, \
    Conv1DTranspose, Activation, GRU, Layer, Dropout, Lambda, UpSampling1D
from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from keras_flops import get_flops
import math
import keras
from tensorflow.keras import optimizers, losses
from keras.callbacks import Callback

# 学习率更新以及调整
def scheduler(epoch):

    if epoch == 0:
        lr = K.get_value(model.optimizer.lr)*10  # keras默认0.001
        K.set_value(model.optimizer.lr, lr)
        print("lr changed to {}".format(lr))
    if epoch != 0 and epoch % 3 == 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * math.pow(0.99, epoch))
        # K.set_value(model.optimizer.lr, lr / (1 + 0.001 * epoch))
        print("lr changed to {}".format(lr))
    return K.get_value(model.optimizer.lr)

class CosineAnnealingScheduler(Callback):
    def __init__(self, T_max, eta_max=0.01, eta_min=0.0001):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(epoch / self.T_max * math.pi))
        setattr(self.model.optimizer, 'lr', lr)
        print('Learning rate for epoch {} is {}'.format(epoch + 1, lr))

# 数据导入
file1 = 'N'

for index in range(10, 11):

    # index = 1

    data1 = np.load('D:/AF_BP/bp/' + file1 + '/train_ppg' + str(index) + '.npy', allow_pickle=True)
    data2 = np.load('D:/AF_BP/bp/' + file1 + '/train_ecg' + str(index) + '.npy', allow_pickle=True)
    data3 = np.load('D:/AF_BP/bp/' + file1 + '/train_abp' + str(index) + '.npy', allow_pickle=True)

    data = Concatenate(axis=1)([data1, data2])

    label1 = np.min(data3, axis=1)
    label2 = np.mean(data3, axis=1)
    label3 = np.max(data3, axis=1)

    val_data1 = np.load('D:/AF_BP/bp/' + file1 + '/test_ppg' + str(index) + '.npy', allow_pickle=True)
    val_data2 = np.load('D:/AF_BP/bp/' + file1 + '/test_ecg' + str(index) + '.npy', allow_pickle=True)
    val_data3 = np.load('D:/AF_BP/bp/' + file1 + '/test_abp' + str(index) + '.npy', allow_pickle=True)

    val_data = Concatenate(axis=1)([val_data1, val_data2])

    val_label1 = np.min(val_data3, axis=1)
    val_label2 = np.mean(val_data3, axis=1)
    val_label3 = np.max(val_data3, axis=1)

    def senet(inputs):

        x = keras.layers.GlobalAveragePooling1D()(inputs)
        y = keras.layers.Dense(int(x.shape[-1])*8, activation='relu')(x)
        z = keras.layers.Dense(int(x.shape[-1]), activation='hard_sigmoid')(y)

        return z

    # padding = 'valid'
    padding = 'same'

    k1, k2, k3, k4 = 3, 5, 7, 9

    def head(inputs, a):

        x0 = Conv1D(filters=a, kernel_size=k1, strides=1, padding=padding)(inputs)
        x0 = BatchNormalization(momentum=0.99, epsilon=0.001)(x0)
        x0 = ELU()(x0)

        x1 = Conv1D(filters=a, kernel_size=k2, strides=1, padding=padding)(inputs)
        x1 = BatchNormalization(momentum=0.99, epsilon=0.001)(x1)
        x1 = ELU()(x1)

        x2 = Conv1D(filters=a, kernel_size=k3, strides=1, padding=padding)(inputs)
        x2 = BatchNormalization(momentum=0.99, epsilon=0.001)(x2)
        x2 = ELU()(x2)

        x3 = Conv1D(filters=a, kernel_size=k4, strides=1, padding=padding)(inputs)
        x3 = BatchNormalization(momentum=0.99, epsilon=0.001)(x3)
        x3 = ELU()(x3)

        w0 = senet(x0)
        w1 = senet(x1)
        w2 = senet(x2)
        w3 = senet(x3)

        W0 = Add()([w1, w2, w3])
        W1 = Add()([w0, w2, w3])
        W2 = Add()([w0, w1, w3])
        W3 = Add()([w0, w1, w2])

        x0 = Multiply()([x0, W0])
        x1 = Multiply()([x1, W1])
        x2 = Multiply()([x2, W2])
        x3 = Multiply()([x3, W3])

        y0 = Conv1D(filters=a, kernel_size=k1, strides=1, padding=padding)(x0)
        y0 = BatchNormalization(momentum=0.99, epsilon=0.001)(y0)
        y0 = ELU()(y0)
        y0 = MaxPooling1D(pool_size=2, strides=2)(y0)

        y1 = Conv1D(filters=a, kernel_size=k2, strides=1, padding=padding)(x1)
        y1 = BatchNormalization(momentum=0.99, epsilon=0.001)(y1)
        y1 = ELU()(y1)
        y1 = MaxPooling1D(pool_size=2, strides=2)(y1)

        y2 = Conv1D(filters=a, kernel_size=k3, strides=1, padding=padding)(x2)
        y2 = BatchNormalization(momentum=0.99, epsilon=0.001)(y2)
        y2 = ELU()(y2)
        y2 = MaxPooling1D(pool_size=2, strides=2)(y2)

        y3 = Conv1D(filters=a, kernel_size=k4, strides=1, padding=padding)(x3)
        y3 = BatchNormalization(momentum=0.99, epsilon=0.001)(y3)
        y3 = ELU()(y3)
        y3 = MaxPooling1D(pool_size=2, strides=2)(y3)

        w0 = senet(y0)
        w1 = senet(y1)
        w2 = senet(y2)
        w3 = senet(y3)

        W0 = Add()([w1, w2, w3])
        W1 = Add()([w0, w2, w3])
        W2 = Add()([w0, w1, w3])
        W3 = Add()([w0, w1, w2])

        y0 = Multiply()([y0, W0])
        y1 = Multiply()([y1, W1])
        y2 = Multiply()([y2, W2])
        y3 = Multiply()([y3, W3])

        return y0, y1, y2, y3

    def DC_Block1(input, k, c):

        conv1 = DepthwiseConv1D(kernel_size=k, strides=1, padding=padding)(input)
        conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
        conv1 = ELU()(conv1)

        conv1 = Conv1D(filters=c, kernel_size=1, strides=1)(conv1)
        conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
        conv1 = ELU()(conv1)

        return conv1

    def bone(inputs0, inputs1, inputs2, inputs3, a):

        x0 = DC_Block1(inputs0, k1, a)
        x1 = DC_Block1(inputs1, k2, a)
        x2 = DC_Block1(inputs2, k3, a)
        x3 = DC_Block1(inputs3, k4, a)

        w0 = senet(x0)
        w1 = senet(x1)
        w2 = senet(x2)
        w3 = senet(x3)

        W0 = Add()([w1, w2, w3])
        W1 = Add()([w0, w2, w3])
        W2 = Add()([w0, w1, w3])
        W3 = Add()([w0, w1, w2])

        x0 = Multiply()([x0, W0])
        x1 = Multiply()([x1, W1])
        x2 = Multiply()([x2, W2])
        x3 = Multiply()([x3, W3])

        z0 = DC_Block1(x0, k1, a)
        z0 = MaxPooling1D(pool_size=2, strides=2)(z0)

        z1 = DC_Block1(x1, k2, a)
        z1 = MaxPooling1D(pool_size=2, strides=2)(z1)

        z2 = DC_Block1(x2, k3, a)
        z2 = MaxPooling1D(pool_size=2, strides=2)(z2)

        z3 = DC_Block1(x3, k4, a)
        z3 = MaxPooling1D(pool_size=2, strides=2)(z3)

        w0 = senet(z0)
        w1 = senet(z1)
        w2 = senet(z2)
        w3 = senet(z3)

        W0 = Add()([w1, w2, w3])
        W1 = Add()([w0, w2, w3])
        W2 = Add()([w0, w1, w3])
        W3 = Add()([w0, w1, w2])

        z0 = Multiply()([z0, W0])
        z1 = Multiply()([z1, W1])
        z2 = Multiply()([z2, W2])
        z3 = Multiply()([z3, W3])

        return z0, z1, z2, z3

    def decd(inputs0, inputs1, inputs2, inputs3, a):

        out0 = Conv1DTranspose(filters=a, kernel_size=3, strides=1)(inputs0)
        out0 = BatchNormalization(momentum=0.99, epsilon=0.001)(out0)
        out0 = ELU()(out0)

        out1 = Conv1DTranspose(filters=a, kernel_size=3, strides=1)(inputs1)
        out1 = BatchNormalization(momentum=0.99, epsilon=0.001)(out1)
        out1 = ELU()(out1)

        out2 = Conv1DTranspose(filters=a, kernel_size=3, strides=1)(inputs2)
        out2 = BatchNormalization(momentum=0.99, epsilon=0.001)(out2)
        out2 = ELU()(out2)

        out3 = Conv1DTranspose(filters=a, kernel_size=3, strides=1)(inputs3)
        out3 = BatchNormalization(momentum=0.99, epsilon=0.001)(out3)
        out3 = ELU()(out3)

        w0 = senet(out0)
        w1 = senet(out1)
        w2 = senet(out2)
        w3 = senet(out3)

        W0 = Add()([w1, w2, w3])
        W1 = Add()([w0, w2, w3])
        W2 = Add()([w0, w1, w3])
        W3 = Add()([w0, w1, w2])

        out0 = Multiply()([out0, W0])
        out1 = Multiply()([out1, W1])
        out2 = Multiply()([out2, W2])
        out3 = Multiply()([out3, W3])

        y0 = Conv1DTranspose(filters=a, kernel_size=3, strides=1)(out0)
        y0 = BatchNormalization(momentum=0.99, epsilon=0.001)(y0)
        y0 = ELU()(y0)

        y1 = Conv1DTranspose(filters=a, kernel_size=3, strides=1)(out1)
        y1 = BatchNormalization(momentum=0.99, epsilon=0.001)(y1)
        y1 = ELU()(y1)

        y2 = Conv1DTranspose(filters=a, kernel_size=3, strides=1)(out2)
        y2 = BatchNormalization(momentum=0.99, epsilon=0.001)(y2)
        y2 = ELU()(y2)

        y3 = Conv1DTranspose(filters=a, kernel_size=3, strides=1)(out3)
        y3 = BatchNormalization(momentum=0.99, epsilon=0.001)(y3)
        y3 = ELU()(y3)

        w0 = senet(y0)
        w1 = senet(y1)
        w2 = senet(y2)
        w3 = senet(y3)

        W0 = Add()([w1, w2, w3])
        W1 = Add()([w0, w2, w3])
        W2 = Add()([w0, w1, w3])
        W3 = Add()([w0, w1, w2])

        y0 = Multiply()([y0, W0])
        y1 = Multiply()([y1, W1])
        y2 = Multiply()([y2, W2])
        y3 = Multiply()([y3, W3])

        return y0, y1, y2, y3

    def models(x0, a=16):

        x1 = head(x0, a=a)

        x2 = bone(x1[0], x1[1], x1[2], x1[3], a=a)

        y0 = MaxPooling1D(pool_size=2, strides=2)(x1[0])
        y1 = MaxPooling1D(pool_size=2, strides=2)(x1[1])
        y2 = MaxPooling1D(pool_size=2, strides=2)(x1[2])
        y3 = MaxPooling1D(pool_size=2, strides=2)(x1[3])

        z0 = Add()([y0, x2[0]])
        z1 = Add()([y1, x2[1]])
        z2 = Add()([y2, x2[2]])
        z3 = Add()([y3, x2[3]])

        z0 = ELU()(z0)
        z1 = ELU()(z1)
        z2 = ELU()(z2)
        z3 = ELU()(z3)

        x3 = bone(z0, z1, z2, z3, a=a)

        y0 = MaxPooling1D(pool_size=2, strides=2)(z0)
        y1 = MaxPooling1D(pool_size=2, strides=2)(z1)
        y2 = MaxPooling1D(pool_size=2, strides=2)(z2)
        y3 = MaxPooling1D(pool_size=2, strides=2)(z3)

        z0 = Add()([y0, x3[0]])
        z1 = Add()([y1, x3[1]])
        z2 = Add()([y2, x3[2]])
        z3 = Add()([y3, x3[3]])

        z0 = ELU()(z0)
        z1 = ELU()(z1)
        z2 = ELU()(z2)
        z3 = ELU()(z3)

        x4 = bone(z0, z1, z2, z3, a=a)

        y0 = MaxPooling1D(pool_size=2, strides=2)(z0)
        y1 = MaxPooling1D(pool_size=2, strides=2)(z1)
        y2 = MaxPooling1D(pool_size=2, strides=2)(z2)
        y3 = MaxPooling1D(pool_size=2, strides=2)(z3)

        z0 = Add()([y0, x4[0]])
        z1 = Add()([y1, x4[1]])
        z2 = Add()([y2, x4[2]])
        z3 = Add()([y3, x4[3]])

        z0 = ELU()(z0)
        z1 = ELU()(z1)
        z2 = ELU()(z2)
        z3 = ELU()(z3)

        x5 = bone(z0, z1, z2, z3, a=a)

        y0 = MaxPooling1D(pool_size=2, strides=2)(z0)
        y1 = MaxPooling1D(pool_size=2, strides=2)(z1)
        y2 = MaxPooling1D(pool_size=2, strides=2)(z2)
        y3 = MaxPooling1D(pool_size=2, strides=2)(z3)

        z0 = Add()([y0, x5[0]])
        z1 = Add()([y1, x5[1]])
        z2 = Add()([y2, x5[2]])
        z3 = Add()([y3, x5[3]])

        z0 = ELU()(z0)
        z1 = ELU()(z1)
        z2 = ELU()(z2)
        z3 = ELU()(z3)

        x6 = bone(z0, z1, z2, z3, a=a)

        y0 = MaxPooling1D(pool_size=2, strides=2)(z0)
        y1 = MaxPooling1D(pool_size=2, strides=2)(z1)
        y2 = MaxPooling1D(pool_size=2, strides=2)(z2)
        y3 = MaxPooling1D(pool_size=2, strides=2)(z3)

        z0 = Add()([y0, x6[0]])
        z1 = Add()([y1, x6[1]])
        z2 = Add()([y2, x6[2]])
        z3 = Add()([y3, x6[3]])

        z0 = ELU()(z0)
        z1 = ELU()(z1)
        z2 = ELU()(z2)
        z3 = ELU()(z3)

        x7 = bone(z0, z1, z2, z3, a=a)

        y0 = MaxPooling1D(pool_size=2, strides=2)(z0)
        y1 = MaxPooling1D(pool_size=2, strides=2)(z1)
        y2 = MaxPooling1D(pool_size=2, strides=2)(z2)
        y3 = MaxPooling1D(pool_size=2, strides=2)(z3)

        z0 = Add()([y0, x7[0]])
        z1 = Add()([y1, x7[1]])
        z2 = Add()([y2, x7[2]])
        z3 = Add()([y3, x7[3]])

        z0 = ELU()(z0)
        z1 = ELU()(z1)
        z2 = ELU()(z2)
        z3 = ELU()(z3)

        x10 = decd(z0, z1, z2, z3, a=a)
        x11 = decd(x10[0], x10[1], x10[2], x10[3], a=a)
        x12 = decd(x11[0], x11[1], x11[2], x11[3], a=a)
        x13 = decd(x12[0], x12[1], x12[2], x12[3], a=a)
        x14 = decd(x13[0], x13[1], x13[2], x13[3], a=a)
        x15 = decd(x14[0], x14[1], x14[2], x14[3], a=a)
        # x16 = decd(x15[0], x15[1], x15[2], x15[3], a=a)
        # x17 = decd(x16[0], x16[1], x16[2], x16[3], a=a)
        # x18 = decd(x17[0], x17[1], x17[2], x17[3], a=a)

        y0 = Flatten()(x15[0])
        y1 = Flatten()(x15[1])
        y2 = Flatten()(x15[2])
        y3 = Flatten()(x15[3])
        out = Add()([y0, y1, y2, y3])

        return out

    def bp_model(inputs):

        x = models(inputs)

        bp1 = Dense(128, activation='linear')(x)
        bp1 = Dropout(0.5)(bp1)
        bp1 = Dense(128, activation='linear')(bp1)
        bp1 = Dropout(0.5)(bp1)

        bp2 = Dense(128, activation='linear')(x)
        bp2 = Dropout(0.5)(bp2)
        bp2 = Dense(128, activation='linear')(bp2)
        bp2 = Dropout(0.5)(bp2)

        bp3 = Dense(128, activation='linear')(x)
        bp3 = Dropout(0.5)(bp3)
        bp3 = Dense(128, activation='linear')(bp3)
        bp3 = Dropout(0.5)(bp3)


        bp1 = Dense(1, activation='linear')(bp1)
        bp2 = Dense(1, activation='linear')(bp2)
        bp3 = Dense(1, activation='linear')(bp3)

        out = Model(inputs=[inputs], outputs=[bp1, bp2, bp3], name="model")

        return out

    inputs = Input(shape=(512, 1))
    model = bp_model(inputs)

    # 计算成本
    model.summary()
    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 6:.05} M")

    # mean_squared_error 均方误差
    # mean_absolute_error 平均绝对误差
    # mean_absolute_percentage_error 平均绝对百分比误差
    # mean_squared_logarithmic_error（MSLE）均方根对数误差

    def root_mean_square_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def mean_square_error(y_true, y_pred):

        return K.square(y_pred - y_true)

    def mean_absolute_error(y_true, y_pred):

        return K.mean(K.abs(y_pred - y_true))

    def huber_loss(y_true, y_pred, delta=1.0):

        error = y_true - y_pred
        quadratic_term = K.minimum(K.abs(error), 0.5 * delta)
        linear_term = K.abs(error) - quadratic_term

        return K.mean(0.5 * K.square(quadratic_term) + delta * linear_term)

    def cos_sim(x, y):

        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)

        return K.mean(x * y, axis=-1, keepdims=True)

    def cos_sim_loss(y_true, y_pred):

        return 1 - cos_sim(y_true, y_pred)

    def loss(y_true, y_pred):

        return (0.6*root_mean_square_error(y_true, y_pred) + 0.4*mean_absolute_error(y_true, y_pred))

    # loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error']
    # loss=[root_mean_square_error, root_mean_square_error, root_mean_square_error]
    # loss=[loss, loss, loss]

    sgd = optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
    # loss_weights = [0.3, 0.5, 0.2]
    model.compile(loss=[loss, loss, loss], loss_weights=[0.3, 0.4, 0.3], optimizer='Adam')

    checkpoint = ModelCheckpoint(filepath='D:/AF_BP/AF_or_BP/model/test_bp' + str(index) + '.hdf5', verbose=2,
                                 monitor='val_loss', mode='min', save_best_only='True')

    reduce_lr = LearningRateScheduler(scheduler)  # 学习率的改变warm_up_lr

    # reduce_lr = CosineAnnealingScheduler(T_max=10, eta_max=0.01, eta_min=0.0001)

    callback_lists = [checkpoint, reduce_lr]
    # [label1, label2, label3]    [val_label1, val_label2, val_label3]
    train_history = model.fit(x=[data],
                              y=[label1, label2, label3], verbose=2,
                              validation_data=[[val_data], [val_label1, val_label2, val_label3]],
                              callbacks=callback_lists,
                              epochs=32, batch_size=512, shuffle=True)