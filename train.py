import tensorflow as tf
from model50 import ResNet
from model3 import build_model

class Trainer(object):
    def __init__(self):
        #self.model = ResNet((24, 72, 1), 1)
        #self.model.build(input_shape=(None, 24, 72, 1))
        self.model = build_model()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                           loss='mse',
                           metrics=['mae'])
        self.batch_size = 256
        self.epochs = 50
        self.weights_path = '/docker/mnt/d/research/D2/resnet/weights/weights.h5'

    def train(self, x_train, y_train, x_val, y_val):
        his = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
        self.model.evaluate(x_val, y_val)
        self.model.save_weights(self.weights_path)

