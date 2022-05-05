from base.base_model import BaseModel
import tensorflow as tf


class MnistModel(BaseModel):
    def __init__(self, config):
        super(MnistModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        inputs = tf.keras.layers.Flatten(input_shape=(28, 28))
        d1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
        dr = tf.keras.layers.Dropout(0.2)(d1)
        out = tf.keras.layers.Dense(10, activation='softmax')(dr)

        pass

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass