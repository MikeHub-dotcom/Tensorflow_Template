import gin
import logging
import wandb
import tensorflow as tf
from absl import app, flags
from utils import dirs, gin_wandb, logger
from data_loader import mnist_loader
from models.mnist_model import MnistModel

import utils.dirs

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')


def main(argv, development=True):

    # User input to name the run
    run_name, tag, group_name = utils.gin_wandb.select_run_names(development)

    # Generate folder structures
    run_paths = utils.dirs.gen_run_folder(run_name)

    # Set loggers
    utils.logger.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # Parse and save the gin-config
    # ToDo: Fix path below
    gin.parse_config_files_and_bindings(['C:/Users/Mikey/PycharmProjects/Tensorflow_Template/configs/config.gin'], [])
    utils.gin_wandb.save_config(run_paths['path_gin'], gin.config_str())

    # Setup WandB
    # ToDo: Enter a project name
    '''wandb.init(project='mnist_test', name=run_name, config=utils.gin_wandb.gin_config_to_readable_dictionary(gin.config._CONFIG),
               group=group_name, tags=tag)'''

    # Setup pipeline
    (ds_train, ds_val), ds_info = mnist_loader.load_mnist_tfds()

    # Load model
    model = MnistModel()
    # ToDo: Check if a custom written train function can be applied to such a model

    # Perform training and evaluation
    # ToDo: Choose metrics according to the labeling of the dataset (e.g. Accuracy, SparseCategoricalAccuracy, BinaryAccuracy, ...)
    # ToDo: Wrap compiling and training into a standalone function
    # ToDo: Save checkpoints and the model at the best point
    if FLAGS.train:
        model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        model.fit(ds_train, validation_data=ds_val, epochs=5)
        print("Evaluating:")
        model.evaluate(ds_val, verbose=2)

    # Perform evaluation only
    else:
        # ToDo: Load model first
        model.evaluate(ds_val, verbose=2)

    # Close WandB run
    #wandb.finish()


if __name__ == "__main__":
    app.run(main)
