import tensorflow as tf
from tensorflow.keras import layers, models 

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (4,8), activation=tf.nn.relu, input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (2,4), activation=tf.nn.relu, padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (2,4), activation=tf.nn.relu, padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation=tf.nn.relu))
    model.add(layers.Dense(1, activation='linear'))
    return model

