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
import time
from timeit import default_timer as timer

from debug_print import debug_print
print = debug_print()

from PIL import ImageFont, ImageDraw, Image

from test_utils import CvPutJaText

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class NotFoundError(Exception):
    pass

def get_unused_dir_num(pdir, pref=None):
    os.makedirs(pdir, exist_ok=True)
    dir_list = os.listdir(pdir)
    for i in range(1000):
        search_dir_name = "" if pref is None else (
            pref + "_" ) + '%03d' % i
        if search_dir_name not in dir_list:
            return os.path.join(pdir, search_dir_name)
    raise NotFoundError('Error')


def draw_img(model, image):

    orig = image.copy()
    # load the trained convolutional neural network

    img_width = model.layers[0].input.shape[2]
    img_height = model.layers[0].input.shape[1]
    try:
        model_image_size = (int(img_width), int(img_height))
        image = cv2.resize(image, model_image_size)
    except:
        pass


    # pre-process the image for classification
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image
    predict = model.predict(image)
    print(predict)

    if class_path is not None:
        class_names = open(class_path, 'r')
        class_names = [line.split('\n')[0] for line in class_names.readlines()]
    else:
        class_names = ['True', 'False']
        predict = [[float(predict[0]), 1.0 - float(predict[0])]]
    pprint(class_names)

    # build labels
    labels = []
    labels.append("{}".format(class_names))
    labels.append(" : ".join(["{:.2f}%".format(100*score) for score in predict[0]]))

    # draw the label on the image
    # output_image = imutils.resize(orig, width=400)
    output_image = orig
    fontPIL = "Dflgs9.ttc"
    font_path = fontPIL
    for i, label in enumerate(labels):
        # cv2.putText(output_image, label, (10, 25 * (i+1)),  cv2.FONT_HERSHEY_SIMPLEX,
        # 	0.7, (0, 255, 0), 2)
        # cv2_putText_2(output_image, label, (10, 25 * (i+1)),  fontPIL,
        # 	10, (0, 255, 0))
        output_image = CvPutJaText.puttext(output_image, label, (10, 25 * (i+1)), font_path, 20, (0, 255, 0))
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", output_image)

        # key = cv2.waitKey(60000)#pauses for 3 seconds before fetching next image
        # if key == 27:#if ESC is pressed, exit loop
        #     cv2.destroyAllWindows()
        #     break
    # def cv2_putText_2(img, text, org, fontFace, fontScale, color):
    return output_image


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=None,
	help="path to trained model model")
ap.add_argument("-i", "--image_path",
	help="path to input image")
ap.add_argument("-v", "--video_path",
	help="path to input video")
ap.add_argument("--binary", action='store_true',
    help="flag for binary classification")

args = vars(ap.parse_args())
if(args["image_path"] is None and args["video_path"] is None):
	ap.error("missing arguments -v / --video_path and -i / --image_path")

if args["binary"]:
    class_path = None
else:
    class_path = os.path.join(os.path.dirname(args["model"]), "classes.txt")
model_path = args["model"] if args["model"] != None else class_path.split(".")[0]+".h5"

print("[INFO] loading network...")
model = load_model(model_path)

output_dir = get_unused_dir_num(pdir="results", pref=None)
os.makedirs(output_dir, exist_ok=True)

if args['video_path']:
    video_path = args["video_path"]
    output_path = os.path.join(output_dir, os.path.basename('out.mp4'))
    vid = cv2.VideoCapture(video_path)
    # vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print(
            "!!! TYPE:",
            type(output_path),
            type(video_FourCC),
            type(video_fps),
            type(video_size))
        print(output_path, video_FourCC, video_fps, video_size)
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    i = 0
    while True:
        return_value, frame = vid.read()
        # print(return_value, frame)
        # print(return_value)
        if return_value == False:
            break
        # frame = Image.fromarray(frame)
        print(type(frame))
        print(frame.shape)
        output_image = draw_img(model, frame)
        print(type(output_image))
        print(output_image.shape)


        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(
            output_image,
            text=fps,
            org=(3, 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.50,
            color=(255, 0, 0),
            thickness=2)


        # break
        # result = np.asarray(output_image)
        # print(return_value)
        # print(frame)
        # print(result)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", output_image)
        if isOutput:
            out.write(output_image)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # cv2.imshow("result", frame)

        # if isOutput:
        #     print('save')
        #     out.write(result)
        # output_path = os.path.join(output_dir, os.path.basename('a.jpg'))
        # print(output_path)
        # cv2.imwrite(output_path, output_image)

        # i +=  1
        # time.sleep(0.1)
        # print(i)
        # sys.exit()
        # break
else:
    image_path = args["image_path"]

    # load the image

    image = cv2.imread(image_path)
    output_image = draw_img(model, image)


    output_dir = get_unused_dir_num(pdir="results", pref=None)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    print(output_path)
    cv2.imwrite(output_path, output_image)