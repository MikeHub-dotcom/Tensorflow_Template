import logging
import tensorflow as tf


def set_loggers(path_log=None, logging_level=0, b_stream=False, b_debug=False):
    """
    ToDo: Fill out below.
    :param path_log:
    :param logging_level:
    :param b_stream:
    :param b_debug:
    :return:
    """
    # Std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # Tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    if path_log:
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)
        logger_tf.addHandler(file_handler)

    # Plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)
