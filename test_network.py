# import the necessary packages
import argparse
import os
from pprint import pprint

import cv2

import imutils
from imutils import paths

import keras
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img

import matplotlib as mpl
import matplotlib.pyplot as plt
import concurrent.futures
import glob

import numpy as np

from test_generator import test_generator
from keras.preprocessing.image import img_to_array
from sklearn import metrics

from utils import get_unused_dir_num
from test_utils import CvPutJaText

import json

fontPIL = "Dflgs9.ttc"
font_path = fontPIL

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

class NotFoundError(Exception):
    pass

def get_unused_result_dir_num(pref):
    os.makedirs('./results', exist_ok=True)
    dir_list = os.listdir(path='./results')
    for i in range(1000):
        search_dir_name = pref + "_" + '%03d' % i
        if search_dir_name not in dir_list:
            return search_dir_name
    raise NotFoundError('Error')


def _read_image_path_and_label_from_test_file(test_file):
    with open(test_file) as f:
        for line in f:
            yield line.split()


def create_polygon_json_from_mask(gray, filename, class_id=None):
    _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line = filename

    for cnt in contours:
        if len(cnt) <= 2:
            continue

        cnt = cnt.flatten()

        line += " "
        line += ",".join(list(map(str, cnt)))
        if class_id is not None:
            line += "," + str(class_id)
    return line

def predict_images(model, class_names, image_path_list=None, image_path=None, imshow=False, predict_dir=None,
                   model_image_size=(28, 28), image_class=None):
    FLAG = predict_dir is None
    if FLAG:
        predict_dir = os.path.join(
            "results", get_unused_result_dir_num("prediction"))

    os.makedirs(predict_dir, exist_ok=True)
    os.makedirs(predict_dir.replace('predictions', 'grad_cam'), exist_ok=True)
    os.makedirs(predict_dir.replace('predictions', 'grad_cam2'), exist_ok=True)
    os.makedirs(predict_dir.replace('predictions', 'heatmap'), exist_ok=True)

    if image_path_list is None:
        image_path_list = list(paths.list_images(image_path)) if os.path.isdir(
            image_path) else [image_path]

    annotations = []

    preds = []
    for path in image_path_list:
        image = cv2.imread(path)
        modelsize_image = cv2.resize(image, model_image_size)
        output_image = imutils.resize(image, width=400)

        # pre-process the image for classification
        orig = cv2.resize(image, model_image_size).astype("float")
        image = orig / 255.0
        orig = cv2.cvtColor(img_to_array(orig), cv2.COLOR_BGR2RGB)
        image = np.expand_dims(img_to_array(image), axis=0)

        # classify the input image
        predict = model.predict(image)
        # pprint(predict)

        ###
        predicted_id = np.argmax(predict[0])
        class_output = model.output[:, predicted_id]

        #  勾配を取得

        for i in range(len(model.layers)):
            if type(model.layers[-i-1]) == keras.layers.convolutional.Conv2D:
                conv_output = model.layers[-i-1].output
                break
        grads = K.gradients(class_output, conv_output)[0]
        # model.inputを入力すると、conv_outputとgradsを出力する関数
        gradient_function = K.function([model.input], [conv_output, grads])

        out, grads_val = gradient_function([image])
        out, grads_val = out[0], grads_val[0]

        # 重みを平均化して、レイヤーのアウトプットに乗じる
        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(out, weights)

        # 画像化してヒートマップにして合成
        cam = cv2.resize(cam, model_image_size,
                         cv2.INTER_LINEAR)  # 画像サイズは299で処理したため
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
        jetcam = (np.float32(heatmap) / 5 + orig / 2)   # もとの画像に合成
        jetcam2 = orig * cam[..., np.newaxis]

        img_basename, img_ext = os.path.splitext(os.path.basename(path))
        extra_output_imgs = [
            (jetcam, "grad_cam"),
            (jetcam2, "grad_cam2"),
            (heatmap, "heatmap"),
        ]
        for extra_output_img, file_suffix in extra_output_imgs:
            save_dir = predict_dir.replace('predictions', file_suffix)
            array_to_img(extra_output_img).resize(
                output_image.shape[1::-1]).save(
                    os.path.join(save_dir,
                                 img_basename + img_ext))
        class_id = image_class if image_class is not None else predicted_id

        ###

        # build labels
        img_labels = []
        img_labels.append("{}".format(class_names))
        img_labels.append(" : ".join(
            ["{:.2f}%".format(100 * score) for score in predict[0]]))

        # draw the label on the image
        for i, img_label in enumerate(img_labels):
            output_image = CvPutJaText.puttext(output_image, img_label, (10, 25 * (i+1)), font_path, 20, (0, 255, 0))


        output_path = os.path.join(predict_dir, os.path.basename(path))
        # print(output_path)
        cv2.imwrite(output_path, output_image)
        preds.append(predict[0])

        if imshow:
            # show the output image
            cv2.imshow("Output", output_image)

            while(cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE) != 0):
                cv2.waitKey(1)

    with open(os.path.join(predict_dir, 'annotation.txt'), mode='w') as f:
        for annotation in annotations:
            f.write(annotation + "\n")


    if FLAG:
        with open(os.path.join(predict_dir, "prediction.csv"), 'w') as f:
            f.write("file,label\n")
            for path, pred in zip(image_path_list, preds):
                f.write("{},{}\n".format(os.path.basename(
                    path), class_names[pred.argmax()]))

    print("Saved predictions to {}/".format(predict_dir))
    return preds


def test_network(model_object, test_data_path, model_path, model_image_size, thre_score):
    model = model_object
    model_dir = os.path.dirname(model_path)
    test_data_dir = os.path.dirname(test_data_path)
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    class_path = os.path.join(model_dir, "classes.txt")
    with open(class_path) as fp:
        class_names = [line.strip() for line in fp]
    CLASS = len(class_names)

    output_dir = os.path.join("results", get_unused_dir_num("results",
                    test_data_path.split('/')[-2] + '_' + config['base_model']))
    os.makedirs(output_dir, exist_ok=True)
    predict_dir = os.path.join(output_dir, "predictions")
    test_path = test_data_path
    contents = _read_image_path_and_label_from_test_file(test_path)
    # result[l1][l2] : the number of images witch is predicted as l1, the true label is l2
    result = np.zeros((CLASS, CLASS), dtype=int)

    labels = []
    predicts = []
    pred_labs = []
    stats = []

    image_path_list = []
    for content in contents:
        image_path_list.append(content[0])
        labels.append(int(content[1]))

    predicts = predict_images(model=model, image_path_list=image_path_list,
                              class_names=class_names, predict_dir=predict_dir, model_image_size=model_image_size)
    predicts = np.array(predicts)
    img_labs = []
    true_labs = []
    u, cnt_labels = np.unique(labels, return_counts=True)
    for i, (pred, lab, image_path) in enumerate(zip(predicts, labels, image_path_list)):
        if thre_score == None:
            argmax = pred.argmax()
            pred_labs.append(argmax)
            true_labs.append(lab)
            img_labs.append(image_path)
            result[argmax][lab] += 1
        else:
            for j, p in enumerate(pred):
                if p > thre_score:
                    pred_labs.append(p)
                    true_labs.append(lab)
                    img_labs.append(image_path)
                    result[j][lab] += 1
    print(result)
    all_sum = len(predicts)
    # col_sum[l] : count of whose true label is l
    col_sum = np.sum(result, axis=0)
    # row_sum[l] : count of whose predicted label is l
    row_sum = np.sum(result, axis=1)
    all_score = []
    all_true = []
    for i in range(CLASS):
        recall = float(result[i][i]) / cnt_labels[i] if col_sum[i] != 0 else -1
        precision = float(result[i][i]) / row_sum[i] if row_sum[i] != 0 else -1
        specificity = float(
            all_sum + result[i][i] - col_sum[i] - row_sum[i]) / (all_sum - col_sum[i])
        f_value = 2 * recall * precision / \
            (recall + precision) if recall != -1 and precision != -1 else -1
        accuracy = (all_sum + 2 * result[i][i] - row_sum[i] -
                    col_sum[i]) / all_sum if all_sum != 0 else -1
        y_score = predicts[:, i]
        y_true = [j == i for j in labels]
        all_score.extend(y_score)
        all_true.extend(y_true)
        print(y_true, y_score)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)

        precision_pr, recall_pr, _ = metrics.precision_recall_curve(
            y_true, y_score)  # tpr is same to recall actually
        auc = metrics.auc(fpr, tpr)
        ap = metrics.average_precision_score(
            y_true, y_score, average='samples')

        C = 0
        thre = 0
        for _tpr, _fpr, threshold in zip(tpr, fpr, thresholds):
            if _tpr - _fpr > C:
                C = _tpr - _fpr
                thre_tpr = _tpr
                thre_fpr = _fpr
                thre = threshold

        # ROC curve
        plt.figure()
        plt.axes().set_aspect('equal')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.fill_between(fpr, 0, tpr, color='y', alpha=0.5)
        plt.plot(fpr, tpr, 'y')
        plt.plot(1 - specificity, recall, color='r',
                 marker='.', markersize=20, alpha=0.7)
        plt.plot(thre_fpr, thre_tpr, color='b',
                 marker='.', markersize=20, alpha=0.7)
        plt.title('ROC curve ' + class_names[i])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "ROC_" + class_names[i] + ".png"))

        # PR curve
        plt.figure()
        plt.axes().set_aspect('equal')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.fill_between(recall_pr, precision_pr, color='r', alpha=0.5)
        plt.plot(recall_pr, precision_pr, 'r')
        plt.title('PR curve ' + class_names[i])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "PR_" + class_names[i] + ".png"))

        stats.append([class_names[i], accuracy, specificity, recall,
                      precision, f_value, auc, ap, thre])

    print("{:<20}\tAccuracy\tSpecificity\tRecall\t\tPrecision\tF value\t\tAUC_ROC\t\tAUC_PR\t\tCutoff value".format(
        "Class name"))
    for stat in stats:
        print("{:<20}".format(stat[0]), end="\t")
        print("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
            stat[1], stat[2], stat[3], stat[4], stat[5], stat[6], stat[7], stat[8]))

    accuracy = float(np.sum([result[i][i] for i in range(CLASS)])) / all_sum
    mAP = metrics.average_precision_score(
            all_true, all_score, average='samples')
    print("\nAccuracy: %f" % accuracy)
    print("\nmAP: %f" % mAP)

    with open(os.path.join(output_dir, "prediction_and_label.txt"), 'w') as f:
        for image_path, pred_lab, label in zip(img_labs, pred_labs, true_labs):
            f.write("%s %d %d\n" % (image_path, pred_lab, label))

    with open(os.path.join(output_dir, "statistics.txt"), 'w') as f:
        f.write("Number_of_classes: %d\nTest_data: %s\n" %
                (CLASS, test_data_path))
        f.write("model_path: %s\n" % (model_path))
        f.write("Accuracy: %f (%d/%d)\n" %
                (accuracy, np.sum([result[i][i] for i in range(CLASS)]), all_sum))
        f.write("mAP: %f\n" % (mAP))
        f.write(
            "Class Accuracy Specificity Recall Precision F-value AUC_ROC AUC_PR Cutoff-value\n")
        for stat in stats:
            f.write("%s %f %f %f %f %f %f %f %f\n" % (
                stat[0], stat[1], stat[2], stat[3], stat[4], stat[5], stat[6], stat[7], stat[8]))

        f.write("result:\n %s\n" % (np.array2string(result)))

    print("Saved predictions, statistics and ROC Curve figures in {}/".format(output_dir))

    return (stats, accuracy)


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)

    ap.add_argument("-m", "--model", required=True,
                    help="path to trained model")
    group.add_argument("-t", "--test_data_path",
                       help="path to test dir or dataset to generate test dir")
    group.add_argument("-i", "--image",
                       help="path to input image")
    group.add_argument("-d", "--directory",
                       help="path to input image directory")
    ap.add_argument("-c", "--image_class", default=None, type=int,
                    help="default class id of the images")
    ap.add_argument("-s", "--image_size", default=28, type=int,
                    help="model image size")
    ap.add_argument("-th", "--thre_score", default=None, type=float,
                    help="thre_score")
    args = vars(ap.parse_args())

    model_path = args["model"]
    image_size = args["image_size"]

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(model_path)

    if args["test_data_path"] is not None:
        test_data_path = args["test_data_path"]
        # if not (os.path.isfile(os.path.join(test_dir, "classes.txt")) and os.path.isfile(os.path.join(test_dir,
        #                                                                                               "test_list.txt"))):
        #     test_dir = test_generator(test_dir)
        test_network(model_object=model, test_data_path=test_data_path, model_path=model_path,
        model_image_size=(image_size, image_size), thre_score=args["thre_score"])
    else:
        # model_image_size = model.get_layer(name="conv2d_1").output_shape[1:3]
        class_path = os.path.join(os.path.dirname(model_path), "classes.txt")
        with open(class_path) as fp:
            class_names = [line.strip() for line in fp]
        arguments = (args["image"], True) if args["image"] is not None else (
            args["directory"], False)
        predict_images(
            model=model, image_path=arguments[0], class_names=class_names, imshow=arguments[1], image_class=args["image_class"])
