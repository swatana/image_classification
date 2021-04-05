import os
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


def test_generator(data_path):
    output_dir = os.path.join("test_data", get_unused_test_data_dir_num())
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    classes = []
    for x in sorted(os.listdir(data_path)):
        if(os.path.isdir(data_path + "/" + x)):
            classes.append(x)

    with open(os.path.join(output_dir, "test.txt"), 'w') as f:
        for i, x in enumerate(classes):
            for path in os.listdir(os.path.join(data_path, x)):
                f.write("%s %d\n" % (os.path.join(data_path, x, path), i))

    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for x in classes:
            f.write("%s\n" % x)
    print("Generated test.txt and classes.txt in {}".format(output_dir))
    return output_dir


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="path to image directory")

    args = vars(ap.parse_args())
    data_path = args["data"]
    test_generator(data_path=data_path)
