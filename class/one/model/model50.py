import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Add, Input
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Res_Block(tf.keras.Model):
    def __init__(self, channel_in = 64, channel_out = 256):
        super().__init__()

        channel = channel_out // 4 #チャンネル数: ボトルネックの最終層で入力層の４倍

        self.bn1 = BatchNormalization()
        self.av1 = Activation(tf.nn.relu)
        self.conv1 = Conv2D(channel, kernel_size=1, strides=1, padding='valid', use_bias=False)
        self.bn2 = BatchNormalization()
        self.av2 = Activation(tf.nn.relu)
        self.conv2 = Conv2D(channel, kernel_size=(3,3), strides=1, padding='same', use_bias=False)
        self.bn3 = BatchNormalization()
        self.av3 = Activation(tf.nn.relu)
        self.conv3 = Conv2D(channel_out, kernel_size=1, strides=1, padding='valid', use_bias=False)
        self.sc = self.shortcut(channel_in, channel_out)
        self.add = Add()

    def shortcut(self, channle_in, channel_out):
        if channle_in != channel_out:
            self.bn_sc1 = BatchNormalization()
            self.conv_sc1 = Conv2D(channel_out, kernel_size=1, strides=1, padding='same', use_bias=False)
            return self.conv_sc1
        else:
            return lambda x:x

    def call(self, x):
        out1 = self.conv1(self.av1(self.bn1(x)))
        out2 = self.conv2(self.av2(self.bn2(out1)))
        out3 = self.conv3(self.av3(self.bn3(out2)))
        short = self.sc(x)
        out4 = self.add([out3, short])
        return out4

class ResNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()

        self.blocks = [
                #Downsampling is performed with a stride of 2
                Conv2D(64, kernel_size=(7,7), strides=2, padding='same', use_bias=False, input_shape=input_shape),#12x36
                MaxPool2D(pool_size=3, strides=2, padding='same'),#6x18x64
                Res_Block(64, 256),#6x18x256
                [Res_Block(256, 256) for _ in range(2)],#6x18x256
                Conv2D(512, kernel_size=1, strides=2),#3x9x512
                [Res_Block(512, 512) for _ in range(4)],#3x9x512
                Conv2D(1024, kernel_size=1, strides=2, use_bias=False),#2x5x1024
                [Res_Block(1024, 1024) for _ in range(6)],#2x5x1024
                Conv2D(2048, kernel_size=1, strides=2, use_bias=False),#1x3x2048
                [Res_Block(2048, 2048) for _ in range(3)],#1x3x2048
                GlobalAveragePooling2D(),#2048
                Dense(1000, activation='relu'),#1000
                Dense(output_dim, activation='softmax')#5
        ]

    def call(self, x):
        for block in self.blocks:
            if isinstance(block, list):
                for b in block:
                    x = b(x)
            else:
                x = block(x)
        return x

    def build(self, input_shape):
        x = Input(shape=(input_shape))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model

