import os
import glob
import random
import argparse


class NotFoundError(Exception):
    pass


def get_unused_test_data_dir_num():
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_data_path, exist_ok=True)
    dir_list = os.listdir(path='./test_data')
    for i in range(1000):
        search_dir_name = '%03d' % i
        if search_dir_name not in dir_list:
            return search_dir_name
    raise NotFoundError('Error')


def test_generator(data_path, test_per):
    output_dir = os.path.join("test_data", get_unused_test_data_dir_num())
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    classes = []
    for x in sorted(os.listdir(data_path)):
        if(os.path.isdir(data_path + "/" + x)):
            classes.append(x)

    with open(os.path.join(output_dir, "train_list.txt"), 'w') as f_train:
        with open(os.path.join(output_dir, "test_list.txt"), 'w') as f_test:
            for i, x in enumerate(classes):

                ext = ["png", "jpg", "gif"]
                pathes = []
                for e in ext:
                    pathes.extend(glob.glob(os.path.join(data_path, x, "*." + e)))
                random.shuffle(pathes)
                test_siz = int(len(pathes) * test_per)
                for path in pathes[:test_siz]:
                    f_test.write("%s %d\n" % (os.path.join(data_path, x, path), i))
                for path in pathes[test_siz:]:
                    f_train.write("%s %d\n" % (os.path.join(data_path, x, path), i))

    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for x in classes:
            f.write("%s\n" % x)
    print("Generated train_list.txt, test_list.txt and classes.txt in {}".format(output_dir))
    return output_dir


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="path to image directory")
    ap.add_argument("-t", "--test_per", default=0.15, type=float,
                    help="test data percentage")

    args = vars(ap.parse_args())
    test_generator(args["data"], args["test_per"])
