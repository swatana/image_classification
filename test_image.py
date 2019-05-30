# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils import paths
import cv2
import random
from pprint import pprint
import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", required=True,
	help="path to class file")
ap.add_argument("-m", "--model", default=None,
	help="path to trained model model")
ap.add_argument("-i", "--image",
	help="path to input image")
ap.add_argument("-d", "--directory",
	help="path to input image directory")

args = vars(ap.parse_args())
if(args["image"] is None and args["directory"] is None):
	ap.error("missing arguments -d / --directory and -i / --images")

class_path=args["class"]
model_path = args["model"] if args["model"] != None else class_path.split(".")[0]+".h5"

# load the image
if(args["image"] is not None):
	image = cv2.imread(args["image"])
else:
	imagePaths = list(paths.list_images(args["directory"]))
	image = cv2.imread(imagePaths[random.randrange(len(imagePaths))])
orig = image.copy()

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(model_path)

model_image_size = model.get_layer(name="conv2d_1").output_shape[1:3]

# pre-process the image for classification
image = cv2.resize(image, model_image_size)
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# classify the input image
predict = model.predict(image)
print(predict)

class_names = open(class_path, 'r')
class_names = [line.split('\n')[0] for line in class_names.readlines()]
pprint(class_names)

# build labels
labels = []
labels.append("{}".format(class_names))
labels.append(" : ".join(["{:.2f}%".format(100*score) for score in predict[0]]))

# draw the label on the image
output = imutils.resize(orig, width=400)
for i, label in enumerate(labels):
	cv2.putText(output, label, (10, 25 * (i+1)),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255, 0), 2)


# show the output image
cv2.imshow("Output", output)

while( cv2.getWindowProperty('Output',cv2.WND_PROP_VISIBLE) != 0):
	cv2.waitKey(1)
