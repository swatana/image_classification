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