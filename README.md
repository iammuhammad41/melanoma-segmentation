# **U-Net Image Segmentation for Medical Images**

This repository contains the code for training a U-Net model for **image segmentation** on medical datasets using **TensorFlow** and **Keras**. The model is built using the **UNet 3+** architecture and employs a variety of techniques including **augmentation**, **model checkpointing**, **early stopping**, and **learning rate scheduling**. The dataset used in this example is from the **ISIC 2017 and PH2** dataset for skin lesion segmentation.

## **Key Features**

* **TensorFlow** and **Keras** based image segmentation model using the **U-Net** architecture.
* Supports **TPU** and **GPU** acceleration.
* Implements **model checkpoints** and **early stopping** to prevent overfitting.
* Incorporates **augmentation** techniques to improve model generalization.
* **Focal loss** and **Dice coefficient** are used as loss functions and evaluation metrics for segmentation.
* The model can be used to segment skin lesions from medical images such as those from the **ISIC** dataset.

## **Requirements**

### Dependencies:

* **TensorFlow 2.x**
* **Keras**
* **Matplotlib**
* **NumPy**
* **OpenCV**
* **Kaggle Datasets API**

You can install the required libraries using the following command:

```bash
pip install tensorflow matplotlib numpy opencv-python kaggle-datasets
```

## **How to Use**

### 1. **Prepare the Dataset**

This code assumes you are using the **ISIC 2017 and PH2** dataset, stored in the following structure:

```
/trainx
/trainy
```

Where:

* `/trainx` contains the input images.
* `/trainy` contains the corresponding segmentation masks.

### 2. **Setting Up the Model**

The model is a **U-Net 3+** architecture with the following configuration:

* **Input size**: 384x384 pixels
* **Backbone**: ResNet (optionally ImageNet pre-trained)
* **Loss function**: Focal Tversky loss
* **Metrics**: Dice coefficient, IOU, Precision, Recall

### 3. **Training the Model**

You can train the model by running the following script:

```python
import tensorflow as tf
from tensorflow import keras
from keras_unet_collection import models, losses
import tensorflow.keras.backend as K

# Set device to TPU or GPU
DEVICE = "TPU"  # or GPU

# Initialize the strategy for distributed training
if DEVICE == "TPU":
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

# Create U-Net model with the given input size and other parameters
input_layer = keras.layers.Input(shape=(384, 384, 3), name="seg_input")
unet_model = models.unet_3plus_2d_base(input_layer, filter_num_down=[32, 64, 128, 256, 512], stack_num_down=2)

# Compile the model
unet_model.compile(optimizer=keras.optimizers.Adam(0.0001), loss=losses.focal_tversky, metrics=[dice_coe])

# Train the model using your dataset
history = unet_model.fit(
    train_dataset_seg,
    epochs=50,
    validation_data=valid_dataset_seg,
    callbacks=[keras.callbacks.ModelCheckpoint("unet_model.h5", save_best_only=True)],
    verbose=2,
)
```

This will train the model using **TPU** or **GPU** (based on the device configuration) for 50 epochs. The training will be logged and saved to `unet_model.h5`.

### 4. **Evaluating the Model**

Once the model is trained, you can evaluate its performance on the test dataset:

```python
test_preds = unet_model.predict(test_dataset_seg)
# Display the predictions
plt.imshow(test_preds[0])
```

### 5. **Prediction and Post-processing**

To make predictions on new data or enhance the predictions, you can use the `enhance_preds` function to threshold the results:

```python
def enhance_preds(img_data, threshold=0.5, dim_x=384, dim_y=384, channels=3):
    preds = unet_model.predict(img_data)
    preds = preds.flatten()
    for i in range(len(preds)):
        preds[i] = 1 if preds[i] > threshold else 0
    return tf.reshape(preds, [dim_x, dim_y, channels])
```

### 6. **Saving Model Weights**

You can save the model weights at any point during the training to resume later:

```python
unet_model.save_weights("unet_model_weights.h5")
```

### 7. **Metrics and Losses**

This model uses the **Dice Coefficient** and **Jaccard Index (IoU)** as the evaluation metrics, and the **Focal Tversky Loss** for training.

```python
def dice_coe(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
```

### **Metrics Functions:**

* `iou(y_true, y_pred)`
* `precision(y_true, y_pred)`
* `recall(y_true, y_pred)`
* `accuracy(y_true, y_pred)`

## **Dataset Paths**

You can set up dataset paths using the following:

```python
IMG_ROOT_PATHS = ["path_to_train_images", "path_to_train_masks"]
```

### **Data Augmentation**

This model also includes data augmentation functionality like random rotations, brightness changes, and flipping for better model generalization.

