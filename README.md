# image-colorization

This project is inspired by the work of [this article](https://arxiv.org/abs/1803.05400).

## Requirements

- PyTorch
- other requirements : `pip install -r requirements.txt`

## Datasets

Here are the datasets that have been used for this project : 
- https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset/
- https://www.kaggle.com/datasets/ashwingupta3012/human-faces
- https://www.kaggle.com/datasets/arnaud58/landscape-pictures

But any set of images can be used here.

## Update parameters

All the parameters are stored in the `conf/params.yml` file.

## Training

- To pre-train a model using ResNet-18 : `python pre-train_generator.py`. The result model is stored in `models/res18-unet-to-device.pt`.

- To train a model, use `python train.py`
The following parameters in `conf/params.yml` should be updated first :
```
dataset:
  train_root_dir: {PATH_TO_TRAIN_DATA}
  n_train_images: {NUMBER_OF_TRAINING_IMAGES}

train:
  epochs: {NUMBER_OF_EPOCHS}
  batch_size: {BATCH_SIZE}
  use_pretrain: {1 or 0}

save:
  D: 'models/{SAVED_DISCRIMINATOR}'
  G: 'models/{SAVED_GENERATOR}'
```

## Testing

- To test a model, use `python test.py`
The following parameters in `conf/params.yml` should be updated first :
```
dataset:
  test_root_dir: {PATH_TO_TEST_DATA}
  n_test_images: {NUMBER_OF_TRAINING_IMAGES}    # -1 for all the images

test:
  pretrained: {1 or 0}      # 1 if pretrained model has been used for the model
  G: 'models/{SAVED_GENERATOR}'
```

## Results

For all pictures :
- top row = original images
- middle row = grayscale images
- bottom row = generated images



Training with general images (5k images) :

<p align="center"> <img src="https://github.com/SamPujade/image-colorization/blob/main/im/general_pretrain_10.png" =250x250 alt="Training with general images"/>

Training with human faces dataset :

<p align="center"> <img src="https://github.com/SamPujade/image-colorization/blob/main/im/faces_pretrain_10.png" =250x250 alt="Training with faces"/>

Training with landscapes dataset :

<p align="center"> <img src="https://github.com/SamPujade/image-colorization/blob/main/im/landscapes_pretrain_10.png" =250x250 alt="Training with landscapes"/>

  
Training with general images (10k images) :

<p align="center"> <img src="https://github.com/SamPujade/image-colorization/blob/main/im/general_10k_pretrain_10.png" =250x250 alt="Training with faces"/>
