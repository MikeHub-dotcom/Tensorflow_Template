import tensorflow as tf

# ToDo: Use MNIST to create the template
# ToDo: Update this template from TF1 to TF2
# ToDo: One config file for one model architecture

'''
Pass config parameters, maybe in a compact and flexible format
Initiate counters (steps, epochs)

-> Checkpoints or saved models wanted?!
Function for saving checkpoints
Function for loading checkpoints
'''

class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        #self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            #self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        #with tf.variable_scope('cur_epoch'):
        #    self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
        #    self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)
        pass

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        #with tf.variable_scope('global_step'):
        #    self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        pass

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
