import os
import random

import numpy as np
from imutils import paths
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from train_utils import get_unused_dir_num
from train_utils import get_unused_log_dir_num
from train_utils import load_images
from train_utils import make_logging_callbacks
from train_utils import modify_base_model

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import json

class SingleLabelNetworkTrainer():
    def __init__(self, train_file_path, val_file_path, classes_file_path, image_width, image_height, model_path, logs_dir=None):

        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.classes_file_path = classes_file_path
        self.logs_dir = logs_dir

        class_names = []
        with open(self.classes_file_path) as classes_fp:
            class_names = [line.strip() for line in classes_fp]
            num_classes = len(class_names)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.image_width = image_width
        self.image_height = image_height
        config_path = os.path.join(logs_dir, "config.json");
        with open(config_path, "w") as f:
            config = {}
            config['base_model'] = model_path
            config['image_width'] = image_width
            config['image_height'] = image_height
            json.dump(config, f)

    def train(self, init_lr, batch_size, num_epochs, full_training,
              is_resuming, binary_classification):
        if self.val_file_path is None:
            x_train, x_test, y_train, y_test, class_names = self.load_dataset()
        else:
            x_train, x_test, y_train, y_test, class_names = self.load_train_val()

        if binary_classification:
            y_train = y_train[:, 1]
            y_test = y_test[:, 1]

        num_classes = len(class_names)

        # construct the image generator for data augmentation
        aug = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode="nearest")

        optimizer = Adam(lr=init_lr, decay=init_lr / num_epochs)
        model = self.compile_model(optimizer, 1 if binary_classification else num_classes, full_training,
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
            monitor='val_loss', patience=100, verbose=1)
        return [
            reduce_lr, early_stopping, *make_logging_callbacks(self.logs_dir)
        ]

    def compile_model(self, optimizer, num_classes, full_training,
                      is_resuming):
        model = modify_base_model(self.base_model, 'sigmoid' if num_classes == 1 else 'softmax', num_classes,
                                  full_training, is_resuming)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy'])
        return model

    def load_dataset(self):
        print("[INFO] loading images...")
        all_labels = []
        img_paths = []

        with open(self.train_file_path) as train_fp:
            for line in train_fp:
                img_path, class_ids = line.split()

                img_paths.append(img_path)
                all_labels.append(int(class_ids))

        with open(self.classes_file_path) as classes_fp:
            class_names = [line.strip() for line in classes_fp]
            num_classes = len(class_names)

        with open(os.path.join(self.logs_dir, "classes.txt"), 'w') as f:
            for item in class_names:
                f.write("%s\n" % item)

        data = load_images(img_paths, self.image_width, self.image_height)
        all_labels = np.array(all_labels)

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        trainX, testX, trainY, testY = train_test_split(
            data, all_labels, test_size=0.25, shuffle=True, random_state=42)
        print(trainY)
        print(testY)

        # convert the labels from integers to vectors
        trainY = to_categorical(trainY, num_classes=num_classes)
        testY = to_categorical(testY, num_classes=num_classes)

        return trainX, testX, trainY, testY, class_names


    def load_train_val(self):
        print("[INFO] loading images...")
        train_paths = []
        train_labels = []
        val_paths = []
        val_labels = []

        with open(self.train_file_path) as train_fp:
            for line in train_fp:
                img_path, class_ids = line.split()
                train_paths.append(img_path)
                train_labels.append(int(class_ids))

        with open(self.train_file_path) as train_fp:
            for line in train_fp:
                img_path, class_ids = line.split()
                val_paths.append(img_path)
                val_labels.append(int(class_ids))

        with open(self.classes_file_path) as classes_fp:
            class_names = [line.strip() for line in classes_fp]
            num_classes = len(class_names)

        with open(os.path.join(self.logs_dir, "classes.txt"), 'w') as f:
            for item in class_names:
                f.write("%s\n" % item)

        trainX = load_images(train_paths, self.image_width, self.image_height)
        trainY = np.array(train_labels)
        testX = load_images(val_paths, self.image_width, self.image_height)
        testY = np.array(val_labels)

        # print(trainX)
        # print(testX)
        # print(trainY)
        # print(testY)

        # convert the labels from integers to vectors
        trainY = to_categorical(trainY, num_classes=num_classes)
        testY = to_categorical(testY, num_classes=num_classes)

        return trainX, testX, trainY, testY, class_names
