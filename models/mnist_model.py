import tensorflow as tf

# Will I need to call save() on it? If so, go with Model
# ToDo: One config file for one model architecture
# ToDo: Give input shape through the config gin
# ToDo: Pass config parameter in a compact format (a second gin or something like that)
#  -> Gin with the same name as the model
#   Save model and best checkpoint after successful training

'''
Pass config parameters, maybe in a compact and flexible format
Initiate counters (steps, epochs) (maybe do that within the train function)

-> Checkpoints or saved models wanted?!
Function for saving checkpoints
Function for loading checkpoints

Using model subclassing after: https://pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/
https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e
'''


class MnistModel(tf.keras.Model):
    """Using Keras model subclassing to create own models"""
    def __init__(self):
        super(MnistModel, self).__init__()

        self.flatten1 = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

        #self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        #self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")

    def __call__(self, inputs, training=False):
        """
        Calls the model on new inputs and returns the outputs as tensors.
        :param inputs: input data tensors (the training/test data)
        :param training: Switch behavior of some layers (e.g. dropout) between training and inference
        :return: output tensor
        """
        out = self.flatten1(inputs)
        out = self.dense1(out)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)
        return out

    # ToDo: Write own training and evaluation routine
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    '''def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            #self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
            # ToDo: Ask on stackoverflow regarding this problem
            loss = tf.keras.losses.SparseCategoricalCrossentropy(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.mae_metric]'''

