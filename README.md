
This repository demonstrates a solution for **segmentation and keypoint detection** on medical image datasets, specifically for the **ISIC 2017** and **PH2 datasets** using the **UNet architecture** in TensorFlow and Keras.

## Table of Contents

1. [Overview](#overview)
2. [Setup and Installation](#setup-and-installation)
3. [Data Preparation](#data-preparation)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Evaluation](#evaluation)
7. [Metrics](#metrics)
8. [Enhancing Predictions](#enhancing-predictions)


## Overview

This project uses **UNet3+** to perform semantic segmentation of skin lesion images and predicts keypoints on the objects. Specifically, it is designed to work with **ISIC 2017** and **PH2 datasets** to train a model for detecting skin lesions and their respective keypoints. The model is trained using **TPUs** and incorporates **TensorFlow**, **Keras**, and **Keras-Unet-Collection** for image segmentation.

## Setup and Installation

To get started, you'll need to install the following dependencies:

```bash
pip install tensorflow
pip install keras-unet-collection
pip install tensorflow-datasets
pip install opencv-python
pip install kaggle-datasets
pip install matplotlib
```

This will install TensorFlow, Keras Unet Collection, and other required libraries.

## Data Preparation

The model accepts **TFRecord** format files for training and evaluation. The `create_seg_tfrecords` function will read the dataset and create TFRecord files, which contain both image and mask annotations.

### TFRecord Generation:

Run the following function to generate the required **TFRecord** files:

```python
create_seg_tfrecords(tfrecord_type="train", SIZE=500, tfrec_roots=None, img_root_paths=IMG_ROOT_PATHS)
```

This will create the **train** and **test** TFRecord files in the directory.

The `get_seg_paths` function handles the generation of paths for **train** and **test** images based on the provided directory structure.

## Model Architecture

We use the **UNet 3+ architecture**, a powerful image segmentation model. Below is the code to initialize and configure the model:

```python
with strategy.scope():
    input_layer = keras.layers.Input(shape=(DIM, DIM, 3), name="seg_input")
    unet_base = base.r2_unet_2d_base(input_layer, filter_num=filter_num, stack_num_down=stack_num_down,
                                     stack_num_up=stack_num_up, recur_num=recur_num, activation="ReLU",
                                     batch_norm=True, pool="max", unpool="nearest", name="res_unet_base")
    unet_output = keras.layers.Conv2D(MASK_CHANNELS, (1, 1), activation="sigmoid")(unet_base)
    unet_model = keras.Model(inputs=[input_layer], outputs=[unet_output])
    unet_model.compile(optimizer=keras.optimizers.Nadam(0.0001),
                       loss=[losses.focal_tversky],
                       metrics=[dice_coe])
```

## Training the Model

The model can be trained using the following code:

```python
history = unet_model.fit(
    train_dataset_seg,
    epochs=50,
    validation_data=valid_dataset_seg,
    callbacks=[keras.callbacks.ModelCheckpoint("r2_unet.h5", monitor='val_loss',
                                               verbose=2, save_best_only=True,
                                               save_weights_only=True, mode='min',
                                               save_freq='epoch'),
               keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
    verbose=2,
)
```

You can monitor the loss and dice coefficient during training. If the model reaches the best validation loss, it will save the model weights.

## Evaluation

After training, the model can be evaluated on the test set as shown below:

```python
unet_model.load_weights("r2_unet.h5")
unet_model.evaluate(test_dataset_seg)
```

## Metrics

We calculate several metrics to evaluate the performance of the segmentation model:

* **Dice Coefficient**: Measures the overlap between predicted and ground truth masks.
* **Jaccard Index**: Measures the intersection over union for segmentation.
* **Accuracy**: Measures the percentage of correct pixel predictions.

Here’s the implementation for **Jaccard Index**:

```python
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)
```

## Enhancing Predictions

You can apply a threshold to enhance the predicted segmentation mask. Here’s an example function:

```python
def enhance_preds(img_data, threshold=0.5, dim_x=384, dim_y=384, channels=3):
    preds = unet_model.predict(img_data)
    preds = preds.flatten()
    for i in range(len(preds)):
        if preds[i] > threshold:
            preds[i] = 1
        else:
            preds[i] = 0 
    return tf.reshape(preds, [dim_x, dim_y, channels])
```


### Troubleshooting:

* Ensure you have a compatible GPU or TPU setup.
* Verify that your dataset is correctly prepared in TFRecord format.
