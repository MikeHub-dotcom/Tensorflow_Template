from base.base_model import BaseModel
import tensorflow as tf


class MnistModel():
    def __init__(self, input_shape):
        # ToDo: Inheritance to super class
        # ToDo: Give input shape through the config gin
        # ToDo: Pass config parameter in a compact format (a second gin or something like that)
        #  -> Gin with the same name as the model

        self.input_shape = input_shape
        self.build_model()


    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        input = tf.keras.Input(self.input_shape)
        out = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
        out = tf.keras.layers.Dense(128, activation='relu')(out)
        out = tf.keras.layers.Dropout(0.2)(out)
        out = tf.keras.layers.Dense(10, activation='softmax')(out)

        return tf.keras.Model(inputs=input, outputs=out, name='mnist_test')
