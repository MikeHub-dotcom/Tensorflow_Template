import gin
import tensorflow as tf
import logging
import datetime
import wandb

"""
ToDo: Making a own trainer with: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
"""


class Trainer(object):

    def __init__(self, model, ds_train, ds_test, run_paths):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = model

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(data, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)



    #@tf.function
    #def test_step(self, images, labels):

    #def train(self):
