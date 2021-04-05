import os

import cv2
import numpy as np
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import img_to_array


class NotFoundError(Exception):
    pass


def get_unused_log_dir_num():
    dir_list = os.listdir(path='./logs')
    for i in range(1000):
        search_dir_name = '%03d' % i
        if search_dir_name not in dir_list:
            return search_dir_name
    raise NotFoundError('Error')


def load_images(img_paths, image_width, image_height):
    """Read and preprocess images"""

    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_width, image_height))
        img = img_to_array(img)
        imgs.append(img)

    return np.array(imgs) / 255.0


def make_logging_callbacks(logs_dir):
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(logs_dir,
                              'model.{epoch:02d}-{val_loss:.2f}.hdf5'),
        verbose=1,
        save_best_only=True,
    )
    csv_logger = CSVLogger(os.path.join(logs_dir, 'training.log'))
    tensor_board = TensorBoard(log_dir=os.path.join(logs_dir, "logs"))
    return [checkpointer, csv_logger, tensor_board]


def modify_base_model(base_model, activation, num_classes, is_fine_tuning,
                      is_resuming):
    """Recreate the last layer of base model.

    Args:
    - is_resuming: If training is resumed then the last layer
    will not be recreated.
    """

    for layer in base_model.layers:
        layer.trainable = not is_fine_tuning

    if is_resuming:
        base_model.layers[-1].trainable = True
        return base_model

    inp = base_model.layers[0].input
    out = base_model.layers[-2].output
    predictions = Dense(num_classes, activation=activation)(out)
    return Model(inp, predictions)
