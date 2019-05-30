# USAGE

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from keras.models import load_model

from pprint import pprint

class NotFoundError(Exception):
    pass

def get_unused_dir_num(pdir, pref=None):
    os.makedirs(pdir, exist_ok=True)
    dir_list = os.listdir(pdir)
    for i in range(1000):
        search_dir_name = '%03d' % i if pref is None else (
            pref + "_" ) + '%03d' % i
        if search_dir_name not in dir_list:
            return os.path.join(pdir, search_dir_name)
    raise NotFoundError('Error')


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model_path",default=None,
	help="path to output model")
args = vars(ap.parse_args())

dataset_path=args["dataset"]
dataset_name = os.path.basename(os.path.dirname(dataset_path))
model_path = args["model_path"] if args["model_path"] != None else None
print (model_path)
# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
labelnames = []
for x in sorted(os.listdir(dataset_path)):
    if(os.path.isdir(dataset_path + "/" + x)):
        labelnames.append(x)

CLASS = len(labelnames)

# initialize the model
print("[INFO] compiling model...")

if(args["model_path"] is not None):
	model = load_model(model_path)
else:
	model = LeNet.build(width=28, height=28, depth=3, classes=CLASS)
pdir =  os.path.join("./model_data", dataset_name)
model_output_dir = get_unused_dir_num(pdir)
os.makedirs(model_output_dir, exist_ok=True)
model_name = os.path.basename(args["model_path"]) if args["model_path"] is not None else "model.h5"
model_output_path = os.path.join(model_output_dir, model_name)
print(model_output_dir)
print(model_name)
print(model_output_path)

# grab the image paths and randomly shuffle thema
imagePaths = sorted(list(paths.list_images(dataset_path)))
random.seed(42)
random.shuffle(imagePaths)
# pprint(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	labelname = imagePath.split(os.path.sep)[-2]
	if labelname in labelnames:
		pass
	else:
		labelnames.append(labelname)
	labels.append(labelnames.index(labelname))

with open(os.path.join(model_output_dir, "classes.txt"), 'w') as f:
    for item in labelnames:
        f.write("%s\n" % item)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=CLASS)
testY = to_categorical(testY, num_classes=CLASS)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
print("[INFO] serialized model name: " + model_output_path)
model.save(model_output_path)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(os.path.join(model_output_dir, "plot.png"))
