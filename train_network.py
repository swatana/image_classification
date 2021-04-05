import argparse

from keras.models import load_model

from single_label_network import SingleLabelNetworkTrainer

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_path",
                        help="Path to base model. Default to InceptionV3")
    parser.add_argument("-b", "--batch_size", default=32, type=int,
                        help="Batch size")
    parser.add_argument("-e", "--epochs", default=10, type=int,
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
    parser.add_argument("--image_size", type=int, default=None,
                        help="image width")
    parser.add_argument("--width", type=int, default=299,
                        help="image width")
    parser.add_argument("--height", type=int, default=299,
                        help="image height")

    args = parser.parse_args()

    width = args.width if args.image_size is None else args.image_size
    height = args.height if args.image_size is None else args.image_size

    batch_size = args.batch_size
    num_epochs = args.epochs
    init_lr = args.learning_rate

    trainer = SingleLabelNetworkTrainer(args.train_file, args.class_file, width, height,
                                        args.logs_dir)
    num_classes = trainer.num_classes
    # initialize the model
    print("[INFO] compiling model...")

    if args.model_path == 'inception_v3':
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(include_top=True, weights='imagenet')
    elif args.model_path:
        base_model = load_model(args.model_path)
    else:
        from lenet import LeNet
        base_model = LeNet.build(width=width, height=height, depth=3, classes=num_classes)

    if init_lr is None:
        init_lr = 1e-3 if args.full_training else 1e-5
    base_model.summary()
    trainer.base_model = base_model

    trainer.train(init_lr, batch_size, num_epochs, not args.full_training,
                  args.resume)


if __name__ == '__main__':
    main()
