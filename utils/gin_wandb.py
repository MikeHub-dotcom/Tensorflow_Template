import tensorflow as tf
# ToDo: Check TensorFlow-specific functionality of the gin.tf module
# https://github.com/google/gin-config


def save_config(path_gin, config):
    """
    Todo: Fill out below
    :param path_gin:
    :param config:
    :return:
    """
    with open(path_gin, 'w') as f_config:
        f_config.write(config)


def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B
    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data["/".join([name, k])] = v

    return data


def get_opt(opt_name):
    """
    ToDo: Fill out below
    Enables the handling of an optimizer-sweep via WandB
    :param opt_name:
    :return:
    """
    if opt_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam()
        return optimizer

    elif opt_name == 'Adadelta':
        optimizer = tf.keras.optimizers.Adadelta()
        return optimizer

    elif opt_name == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad()
        return optimizer

    elif opt_name == 'Adamax':
        optimizer = tf.keras.optimizers.Adamax()
        return optimizer

    elif opt_name == 'Ftrl':
        optimizer = tf.keras.optimizers.Ftrl()
        return optimizer

    elif opt_name == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam()
        return optimizer

    elif opt_name == 'Rmsprop':
        optimizer = tf.keras.optimizers.RMSprop()
        return optimizer

    elif opt_name == 'Sgd':
        optimizer = tf.keras.optimizers.SGD()
        return optimizer

    else:
        print("Invalid optimizer name given within the gin-config. Exiting...")
        exit()


def select_run_names():
    """
    Lets the user choose run-name, group-name and tag for a run
    :return: the variables including the user's input for the respective value
    """
    run_name = input("Specify the name of the run: ")
    group_name = input("Specify the group of the run: ")
    tag_choice = input("Select a tag: (1) - test, (2) - experimental, (3) - serious: ")
    tag = []

    if tag_choice == '1':
        tag.append('Test')

    elif tag_choice == '2':
        tag.append('Experimental')

    elif tag_choice == '3':
        tag.append('Serious')

    elif tag_choice == '':
        print("No tag selected, continuing...")

    return run_name, tag, group_name
