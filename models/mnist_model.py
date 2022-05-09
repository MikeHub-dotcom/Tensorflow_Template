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

    def call(self, inputs, training=False):
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

