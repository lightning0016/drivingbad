import random

import numpy as np
import tensorflow as tf
from keras import Input, Model, losses
from keras.layers import Dense, Multiply, BatchNormalization

x_len = 10
need_print = None


def load_data():
    rs_x = []
    rs_y = []
    for i in range(1, 100000):
        rd_num = random.randint(0, 100000)
        x = np.ones([1, 10])
        x[0, random.randint(0, 9)] = rd_num
        # x[0, random.randint(0, 9)] = random.randint(0,100000)
        rs_x.append(x)
        rs_y.append([[rd_num]])
    return np.array(rs_x), np.array(rs_y)


def model_att():
    x = Input(shape=(1, 10))
    y = Input(shape=(1, 1))
    att_layer = BatchNormalization()(Dense(128, activation='relu')(x))
    att_layer = Dense(256, activation='relu')(att_layer)
    att = Dense(10, activation='softmax')(att_layer)
    hx = Multiply()([x, att])
    # need_print = [x, att]
    hx = Dense(256, activation='relu')(hx)
    hx = Dense(1, activation='relu')(hx)
    ############attention###################

    model = Model(x, hx)

    def myloss(pt, pp):
        pp = tf.Print(pp, [x, '----------', att, pt[0], pp[0]], summarize=10)
        return losses.mean_squared_error(pt, pp)

    model.compile(optimizer='adam', loss=myloss)
    model.summary()
    return model


if __name__ == '__main__':
    rs_x, rs_y = load_data()
    md = model_att()
    md.fit(rs_x, rs_y, batch_size=100, epochs=200)
