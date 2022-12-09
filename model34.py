import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Add, Input, ZeroPadding2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPool2D, AveragePooling2D

def identity_block(x, filter):
    x_skip = x
    x = Conv2D(filter, (3,3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filter, (3,3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

def conv_block(x, filter):
    x_skip = x
    x = Conv2D(filter, (3,3), padding='same', strides=(2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(filter, (3,3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x_skip = Conv2D(filter, (1,1), strides=(2,2))(x_skip)
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

def ResNet19(shape=(32,32,3), classes=10):
    x_input = Input(shape)
    x = ZeroPadding2D((3,3))(x_input)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    block_layers = [3,4,6,3]
    filter_size = 64
    for i in range(len(block_layers)):
        if i == 0:
            for _ in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            filter_size = filter_size*2
            x = conv_block(x, filter_size)
            for _ in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    x = AveragePooling2D((2,2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs=x_input, outputs=x, name='ResNet19')
    return model
