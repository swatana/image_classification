import argparse
import os

from tensorflow.keras.models import load_model

from single_label_network import SingleLabelNetworkTrainer
from train_utils import get_unused_dir_num
from tensorflow.keras.layers import Input
import json

# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_path", default='lenet',
                        help="Path to base model. Default to InceptionV3")
    parser.add_argument("-b", "--batch_size", default=32, type=int,
                        help="Batch size")
    parser.add_argument("-e", "--epochs", default=100, type=int,
                        help="Number of epochs")
    parser.add_argument("-r", "--learning_rate", type=float,
                        help="Learning rate")
    parser.add_argument("-f", "--full_training", action="store_true",
                        help="Train all layers. Default to fine tuning only.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training (won't recreate last layers)")
    parser.add_argument("-l", "--logs_dir",
                        help="Path to output logs")
    parser.add_argument("-t", "--train_file", type=str,
                        help="Path to train file")
    parser.add_argument("-s", "--image_size", type=int, default=None,
                        help="image width")
    parser.add_argument("-iw", "--image_width", type=int, default=299,
                        help="image width")
    parser.add_argument("-ih", "--image_height", type=int, default=299,
                        help="image height")
    parser.add_argument("-ca", "--classifier_activation", type=str, default='softmax',
                        help="activation")

    args = parser.parse_args()

    image_width = args.image_width if args.image_size is None else args.image_size
    image_height = args.image_height if args.image_size is None else args.image_size

    batch_size = args.batch_size
    num_epochs = args.epochs
    init_lr = args.learning_rate
    train_file_path = args.train_file
    model_path = args.model_path
    logs_dir = args.logs_dir
    full_training = args.full_training
    resume = args.resume
    classifier_activation = args.classifier_activation
    class_file_path = os.path.join(os.path.dirname(train_file_path), "classes.txt")


    if logs_dir is None:
        logs_dir = os.path.join("logs", get_unused_dir_num("logs",train_file_path.split('/')[-2] + '_' + model_path))
    os.makedirs(logs_dir, exist_ok=True)

    config_path = os.path.join(logs_dir, "config.json");
    with open(config_path, "w") as f:
        config = {}
        config['base_model'] = model_path
        config['image_width'] = image_width
        config['image_height'] = image_height
        config['classifier_activation'] = classifier_activation
        json.dump(config, f)

    trainer = SingleLabelNetworkTrainer(train_file_path, class_file_path, image_width, image_height,
                                        model_path, logs_dir)
    num_classes = trainer.num_classes
    # initialize the model
    print("[INFO] compiling model...")

    if model_path == 'lenet':
        from lenet import LeNet
        base_model = LeNet.build(height=image_height, width=image_width, depth=3, classes=num_classes, classifier_activation=classifier_activation)
    elif model_path == 'inception_v3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(include_top=True, weights='imagenet', input_tensor=Input(shape=(image_height, image_width, 3)))
    elif model_path == 'EfficientNetB7':
        from tensorflow.keras.applications.efficientnet import EfficientNetB7
        base_model = EfficientNetB7(include_top=True, weights='imagenet', input_tensor=Input(shape=(image_height, image_width, 3)))
    elif model_path == 'mobilenet':
        from tensorflow.keras.applications.mobilenet import MobileNet
        base_model = MobileNet(include_top=True, weights='imagenet')
    else:
        base_model = load_model(model_path)

    if init_lr is None:
        init_lr = 1e-3 if full_training else 1e-5
    base_model.summary()
    trainer.base_model = base_model

    trainer.train(init_lr, batch_size, num_epochs, not full_training,
                  resume)


if __name__ == '__main__':
    main()
