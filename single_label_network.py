import os
import random

import numpy as np
from imutils import paths
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from train_utils import get_unused_log_dir_num
from train_utils import load_images
from train_utils import make_logging_callbacks
from train_utils import modify_base_model

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

class SingleLabelNetworkTrainer():
    def __init__(self, dataset_path, image_size, logs_dir=None):

        if dataset_path[-1] == '/':
            dataset_path = dataset_path[:-1]
        self.dataset_path = dataset_path

        if logs_dir is None:
            logs_dir = os.path.join("logs", get_unused_log_dir_num())
        os.makedirs(logs_dir, exist_ok=True)
        self.logs_dir = logs_dir

        class_names = []
        for x in sorted(os.listdir(self.dataset_path)):
            if os.path.isdir(self.dataset_path + "/" + x):
                class_names.append(x)
        num_classes = len(class_names)
        self.num_classes = num_classes
        self.image_size = image_size

    def train(self, init_lr, batch_size, num_epochs, is_fine_tuning,
              is_resuming):
        x_train, x_test, y_train, y_test, class_names = self.load_dataset()
        num_classes = len(class_names)

        # construct the image generator for data augmentation
        aug = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest")

        optimizer = Adam(lr=init_lr, decay=init_lr / num_epochs)
        model = self.compile_model(optimizer, num_classes, is_fine_tuning,
                                   is_resuming)
        model.summary()

        # train the network
        print("[INFO] training network...")
        history = model.fit_generator(
            aug.flow(x_train, y_train, batch_size=batch_size),
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // batch_size,
            epochs=num_epochs,
            # validation_steps=800,
            verbose=1,
            callbacks=self.make_callbacks(),
        )

        return history

    def make_callbacks(self):
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, verbose=1)
        return [
            reduce_lr, early_stopping, *make_logging_callbacks(self.logs_dir)
        ]

    def compile_model(self, optimizer, num_classes, is_fine_tuning,
                      is_resuming):
        model = modify_base_model(self.base_model, 'softmax', num_classes,
                                  is_fine_tuning, is_resuming)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy'])
        return model

    def load_dataset(self):
        print("[INFO] loading images...")
        labels = []
        class_names = []
        for x in sorted(os.listdir(self.dataset_path)):
            if os.path.isdir(self.dataset_path + "/" + x):
                class_names.append(x)

        num_classes = len(class_names)

        # grab the image paths and randomly shuffle them
        imagePaths = sorted(list(paths.list_images(self.dataset_path)))
        random.seed(42)
        random.shuffle(imagePaths)

        print("Processing {} images...".format(len(imagePaths)))
        # loop over the input images
        for imagePath in imagePaths:
            # extract the class label from the image path and update the
            # labels list
            labelname = imagePath.split(os.path.sep)[-2]
            labels.append(class_names.index(labelname))

        with open(os.path.join(self.logs_dir, "classes.txt"), 'w') as f:
            for item in class_names:
                f.write("%s\n" % item)

        with open(os.path.join(self.logs_dir, "config.txt"), 'w') as f:
            f.write("%s\n" % self.dataset_path)

        data = load_images(imagePaths, self.image_size)
        labels = np.array(labels)

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        trainX, testX, trainY, testY = train_test_split(
            data, labels, test_size=0.25, random_state=42)

        # convert the labels from integers to vectors
        trainY = to_categorical(trainY, num_classes=num_classes)
        testY = to_categorical(testY, num_classes=num_classes)

        return trainX, testX, trainY, testY, class_names
