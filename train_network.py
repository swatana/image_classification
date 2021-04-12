import argparse
import os

from keras.models import load_model

from single_label_network import SingleLabelNetworkTrainer
from train_utils import get_unused_dir_num
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
    parser.add_argument("-c", "--class_file", type=str,
                        help="Path to class names file")
    parser.add_argument("-s", "--image_size", type=int, default=None,
                        help="image width")
    parser.add_argument("--image_width", type=int, default=299,
                        help="image width")
    parser.add_argument("--image_height", type=int, default=299,
                        help="image height")

    args = parser.parse_args()

    image_width = args.image_width if args.image_size is None else args.image_size
    image_height = args.image_height if args.image_size is None else args.image_size

    batch_size = args.batch_size
    num_epochs = args.epochs
    init_lr = args.learning_rate
    train_file_path = args.train_file
    model_path = args.model_path
    logs_dir = args.logs_dir


    if logs_dir is None:
        logs_dir = os.path.join("logs", get_unused_dir_num("logs",train_file_path.split('/')[-2] + '_' + model_path))
    os.makedirs(logs_dir, exist_ok=True)

    trainer = SingleLabelNetworkTrainer(train_file_path, args.class_file, image_width, image_height,
                                        model_path, logs_dir)
    num_classes = trainer.num_classes
    # initialize the model
    print("[INFO] compiling model...")

    if args.model_path == 'lenet':
        from lenet import LeNet
        base_model = LeNet.build(width=image_width, height=image_height, depth=3, classes=num_classes)
    elif args.model_path == 'inception_v3':
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(include_top=True, weights='imagenet')
    elif args.model_path == 'mobilenet':
        from keras.applications.mobilenet import MobileNet
        base_model = MobileNet(include_top=True, weights='imagenet')
    else:
        base_model = load_model(args.model_path)

    if init_lr is None:
        init_lr = 1e-3 if args.full_training else 1e-5
    base_model.summary()
    trainer.base_model = base_model

    trainer.train(init_lr, batch_size, num_epochs, not args.full_training,
                  args.resume)


if __name__ == '__main__':
    main()
