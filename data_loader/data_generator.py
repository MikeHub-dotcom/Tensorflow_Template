import numpy as np

# ToDo: Code input pipeline with tf.data and improve performance with the TF-guide. Make TF-Record files.
# https://www.tensorflow.org/guide/data
# https://www.tensorflow.org/guide/data_performance

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
