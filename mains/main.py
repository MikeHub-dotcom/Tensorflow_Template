import gin
import logging
import wandb
from absl import app, flags
from utils import dirs, gin_wandb, logger

import utils.dirs

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')


def main(argv, development=True):
    # ToDo: Add wandb.init(...) to the evaluation script
    # User input to name the run
    if development:
        run_name, tag, group_name = '', ['dev'], 'dev'
    else:
        run_name, tag, group_name = utils.gin_wandb.select_run_names()

    # Generate folder structures
    run_paths = utils.dirs.gen_run_folder(run_name)

    # Set loggers
    utils.logger.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # Parse and save the gin-config
    gin.parse_config_files_and_bindings(['/home/mikey/PycharmProjects/Tensorflow_custom_barebone/configs/config.gin'], [])
    utils.gin_wandb.save_config(run_paths['path_gin'], gin.config_str())

    # Setup WandB
    # ToDo: Enter a project name
    wandb.init(project='', name=run_name, config=utils.gin_wandb.gin_config_to_readable_dictionary(gin.config._CONFIG),
               group=group_name, tags=tag)

    # Setup pipeline
    # ...

    # Load model
    # ...

    # Perform training and evaluation
    # ...

    # Perform evaluation only
    # ...

    # Close WandB run
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
