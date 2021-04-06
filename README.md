# object-classification

## How to Use

## Prepare package
```
pip3 install -r requirements.txt
```

### Train model
```
python3 train_image.py -d=data/vegetables/
```

### Process model on console
```
python3 test_image.py -m=model_data/vegetables/000/model.h5 -c=model_data/vegetables/000/classes.txt -i=data/vegetables/tomato/18bdb508fd44bff5df656448c57e00cb4a751572.jpg
```


## Test generator
### On terminal
```
python3 test_generator.py -d [DATASET_PATH]

python3 test_generator.py -d data/vegetables
```

## Train network
### On terminal
```
python3 train_network.py -d [DATASET DIR]

python3 train_network.py -t test_data/vegetables_000/train_list.txt -c test_data/vegetables_000/classes.txt -f --image_size 28
```

## Test network
### Evaluate a model by applying it to dataset with labels.
```
python3 test_network.py -m [MODEL_PATH] -t [TEST_DIR]

python3 test_network.py -m logs/005/vegetables_model.21-0.38.hdf5 -t test_data/001
```