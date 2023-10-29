"""
AUTHORS: Altieri J. , Mazzini V.

This program will train two models to perform cephalometric landmark detection.
The first model is a U-Net to do a coarse detection of each landmark ROI;
The second model is a ResNet50 which will perform the actual keypoint detection.

A data augmentation process is also possible and present as a function, beware that this might be time-consuming.
"""
import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pickle

# Filter out TFA warning
#import warnings
#warnings.filterwarnings("ignore", message="TFA has entered a minimal maintenance and release mode", category=Warning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ignore TF unsupported NUMA warnings
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import tensorflow_addons as tfa

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)



######################################## HYPERPARAMETERS AND OTHER OPTIONS ########################################
# Choose if the model should be trained
TRAINING = False

# Choose the name of the saved dataset
MODEL_NAME = "unet_100epochs.keras"
TRAINING_HISTORY = "training_history.pickle"

# Training hyperparameters
BATCH_SIZE = 3
EPOCHS = 100
LRELU_ALPHA = 0.01 # alpha coefficient of LeakyReLU
L2REG = 0.0001 # kernel regularizer coefficient

# Adam hyperparameters
LR=0.001
BETA_1=0.9
BETA_2=0.999
EPSILON=1e-07

# ReduceLROnPlateau hyperparameters
RLR_FACTOR = 0.2
RLR_PATIENCE = 5
RLR_MIN = 0.001

# Chose the fixed image-label pair you want to train on
fixed_image_path = "/mnt/c/Users/jacop/Desktop/DL_Project/1124.jpg"
fixed_label_path = "/mnt/c/Users/jacop/Desktop/DL_Project/1124.txt"


######################################## Importing Dataset #######################################################
print("Started dataset loading...")

input_path = "/mnt/c/Users/jacop/Desktop/DL_Project/processed_dataset/" #if in wsl
#input_path = "/mnt/c/Users/vitto/Desktop/DL project/DL project github/processed_dataset/" #if in wsl
#input_path = "C:/Users/vitto/Desktop/DL project/DL project github/processed_dataset/"  # if in windows
#input_path = "C:/Users/jacop/Desktop/DL_Project/processed_dataset/"  # if in windows

# =========== Images =========== #

def process_image(x):
    """    Reads the file as a sequence of bytes and converts it
    into a grayscale JPG image normalized in [0,1]

    Args:
        x (str): path to the image

    Returns:
        img: processed image
    """
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.squeeze(img, axis=-1)
    img = img / 255
    return img

# Create train, test, val datasets and process the images
train_images = tf.data.Dataset.list_files(input_path + "/images/train/*.jpg", shuffle=False)
train_images = train_images.map(process_image)
test_images = tf.data.Dataset.list_files(input_path + "/images/test/*.jpg", shuffle=False)
test_images = test_images.map(process_image)
val_images = tf.data.Dataset.list_files(input_path + "/images/val/*.jpg", shuffle=False)
val_images = val_images.map(process_image)

# Processing the fixed image
fixed_image = tf.data.Dataset.list_files(fixed_image_path, shuffle=False)
fixed_image = fixed_image.map(process_image)

# =========== Labels =========== #
def load_labels(path):
    """
    Load landmark labels from a specified file and return them as a NumPy array.

    Args:
        path (str): The file path to the label data.

    Returns:
        numpy.ndarray: An array containing normalized landmark coordinates.
    """
    landmarks = []
    with open(path.numpy(), "r", encoding="utf-8") as file:
        next(file, None)
        for line in file:
            columns = line.strip().split("\t")
            x_value = float(columns[1]) / 256
            y_value = float(columns[2]) / 256
            landmarks.extend([x_value, y_value])
    return np.array(landmarks)

# Create train, test and val label dataset and load their labels
train_labels = tf.data.Dataset.list_files(input_path + "/labels/train/*.txt", shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))
test_labels = tf.data.Dataset.list_files(input_path + "/labels/test/*.txt", shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))
val_labels = tf.data.Dataset.list_files(input_path + "/labels/val/*.txt", shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

# Processing the fixed image's labels
fixed_label = tf.data.Dataset.list_files(fixed_label_path, shuffle=False)
fixed_label = fixed_label.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

# =========== Combine Images and Labels =========== #
train_dataset = tf.data.Dataset.zip((train_images, train_labels)).shuffle(1000)
test_dataset = tf.data.Dataset.zip((test_images, test_labels)).shuffle(1000)
val_dataset = tf.data.Dataset.zip((val_images, val_labels)).shuffle(1000)

fixed_dataset = tf.data.Dataset.zip((fixed_image,fixed_label))




######################################## U-NET #######################################################

# =========== Generators =========== #
print("Creating generators...")
  
# Extract only the images
train_images_only = train_dataset.map(lambda img, lbl: img)
test_images_only = test_dataset.map(lambda img, lbl: img)
val_images_only = val_dataset.map(lambda img, lbl: img)
fixed_image_only = fixed_dataset.map(lambda img, lbl: img)

train_list = list(train_images_only)
val_list = list(val_images_only)
test_list = list(test_images_only)

fixed_image = list(fixed_image_only)[0]

# Create a generator function to yield the combined image tensor for training
def train_image_generator():
    """
    Generate training image pairs with fixed and moving images.

    Yields:
        Tuple[tf.Tensor, tf.Tensor]: A pair of combined image tensors with shapes
        (256, 256, 2) representing fixed and moving images, and (256, 256) representing
        the fixed image.
    """
    for moving_image in train_list:
        # Combine fixed and moving images into a single tensor
        combined_image = tf.stack([fixed_image,moving_image],-1)
        yield (combined_image, fixed_image)


# Create a TensorFlow dataset from the generator
train_images_dataset = tf.data.Dataset.from_generator(
    train_image_generator,
    output_signature=(tf.TensorSpec(shape=(256, 256, 2), dtype=tf.float32),
                      tf.TensorSpec(shape=(256, 256), dtype=tf.float32))
    )
train_images_dataset = train_images_dataset.batch(BATCH_SIZE)


# Repeat the process for validation and test
def val_image_generator():
    """
    Generate validation image pairs with fixed and moving images.

    Yields:
        Tuple[tf.Tensor, tf.Tensor]: A pair of combined image tensors with shapes
        (256, 256, 2) representing fixed and moving images, and (256, 256) representing
        the fixed image.
    """
    for moving_image in val_list:
        combined_image = tf.stack([fixed_image, moving_image], -1)
        yield (combined_image, fixed_image)

val_images_dataset = tf.data.Dataset.from_generator(
    val_image_generator,
    output_signature=(
        tf.TensorSpec(shape=(256, 256, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256), dtype=tf.float32)
    )
)
val_images_dataset = val_images_dataset.batch(BATCH_SIZE)


def test_image_generator():
    """
    Generate test image pairs with fixed and moving images.

    Yields:
        Tuple[tf.Tensor, tf.Tensor]: A pair of combined image tensors with shapes
        (256, 256, 2) representing fixed and moving images, and (256, 256) representing
        the fixed image.
    """
    for moving_image in test_list:
        combined_image = tf.stack([fixed_image, moving_image], -1)
        yield (combined_image, fixed_image)

test_images_dataset = tf.data.Dataset.from_generator(
    test_image_generator,
    output_signature=(
        tf.TensorSpec(shape=(256, 256, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256), dtype=tf.float32)
    )
)
test_images_dataset = test_images_dataset.batch(BATCH_SIZE)


# =========== Network Architecture =========== #
# a: activation, c: convolution,
# p: pooling, u: upconvolution, bn: batch normalization

input_shape = (256, 256, 2) # Two images, one fixed and one moving
input = tf.keras.layers.Input(shape=input_shape)

### Downsampling path ###

# Applying Leaky ReLU activation function with an alpha coefficient
a1 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(input)

# Applying a 2D convolution layer with 64 filters of size 3x3
c1 = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2REG))(a1)

# Applying batch normalization
bn1 = tf.keras.layers.BatchNormalization()(c1)

# Applying a Dropout layer with a dropout rate of 10%
c1 = tf.keras.layers.Dropout(0.1)(c1)

# Applying max-pooling with a pooling window of size 2x2
p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1)

a2 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(p1)
c2 = tf.keras.layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2REG))(a2)
bn2 = tf.keras.layers.BatchNormalization()(c2)
c2 = tf.keras.layers.Dropout(0.1)(c2)
p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c2)

a3 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(p2)
c3 = tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2REG))(a3)
bn3 = tf.keras.layers.BatchNormalization()(c3)
c3 = tf.keras.layers.Dropout(0.2)(c3)
p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c3)

a4 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(p3)
c4 = tf.keras.layers.Conv2D(512, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2REG))(a4)
bn4 = tf.keras.layers.BatchNormalization()(c4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

a5 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(p4)
c5 = tf.keras.layers.Conv2D(1024, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2REG))(a5)
bn5 = tf.keras.layers.BatchNormalization()(c5)
c5 = tf.keras.layers.Dropout(0.3)(c5)

### Upsampling path ###

# Decode the output of the encoder using 2D convolution transpose
u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = tf.keras.layers.concatenate([u6, c4])  # Concatenate with previous encoder layers
a6 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(u6)
c6 = tf.keras.layers.Conv2D(512, 3, padding="same", kernel_initializer="he_normal")(a6)
c6 = tf.keras.layers.Dropout(0.3)(c6)

u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
a7 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(u7)
c7 = tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer="he_normal")(a7)
c7 = tf.keras.layers.Dropout(0.2)(c7)

u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
a8 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(u8)
c8 = tf.keras.layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal")(a8)
c8 = tf.keras.layers.Dropout(0.2)(c8)

u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
a9 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(u9)
c9 = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal")(a9)
c9 = tf.keras.layers.Dropout(0.1)(c9)

c10 = tf.keras.layers.Conv2D(2, 1, padding="same", kernel_initializer="he_normal")(c9)

# Create the displacement tensor
displacement_tensor = tf.keras.layers.Conv2D(
    2, kernel_size=3, activation='linear', padding="same", name="disp"
)(c10)


def extract_moving_img(input):
    """
    Extracts the moving image from the input tensor.

    Args:
        input (tf.Tensor): A 3D tensor with shape (256, 256, 2) containing fixed and moving images.

    Returns:
        tf.Tensor: A 2D tensor with shape (256, 256) representing the extracted moving image.
    """
    return input[:, :, :, 1:2]

moving_image = tf.keras.layers.Lambda(extract_moving_img)(input)

def apply_deformation(inputs):
    """
    Apply a deformation transformation to an image using a displacement tensor.

    Args:
        inputs (list): A list containing two TensorFlow tensors, 'image' and 'displacement_tensor'.
        
    Returns:
        tf.Tensor: The deformed image resulting from applying the displacement tensor to the input image.
    """
    image, displacement_tensor = inputs
    deformed_image = tfa.image.dense_image_warp(image, displacement_tensor)
    return deformed_image

def_image = tf.keras.layers.Lambda(apply_deformation)([moving_image, displacement_tensor])

output = def_image


# =========== Model Initialization =========== #
unet = tf.keras.Model(inputs=input, outputs=output)

# Define the callback to reduce the learning rate when the validation loss stops improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=RLR_FACTOR,
                              patience=RLR_PATIENCE, min_lr=RLR_MIN, amsgrad=False)


adam = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)

unet.compile(optimizer=adam, loss='mse', metrics='mse')


# =========== Model Training =========== #
if TRAINING:
    print("training the model...")
    history = unet.fit(train_images_dataset, epochs=EPOCHS, validation_data=val_images_dataset, callbacks=[reduce_lr])
    
    # save the model
    model_name = MODEL_NAME
    training_hist_name = TRAINING_HISTORY
    unet.save(model_name)
    print('Model saved as', model_name)
    # Save the history via pickle
    with open(training_hist_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
# Load a pretrained model
unet = tf.keras.models.load_model(MODEL_NAME,safe_mode=False)
# Load the training history 
with open(TRAINING_HISTORY, "rb") as file_pi:
    history = pickle.load(file_pi)
    
# Plot training & validation accuracy values
plt.plot(history['mse'])
plt.plot(history['val_mse'])
plt.title('MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('mse.png')

# Plot training & validation loss values
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png')


# =========== Model Testing =========== #
# Evaluate the model on the test dataset
print("Evaluate on test data")
results = unet.evaluate(test_images_dataset)
print("Test loss, Test accuracy:", results)

# plotting a deformation example

test_image_list = list(test_images_only)
test_image = test_image_list[3]
test_feed = tf.stack([fixed_image, test_image],-1)
test_feed = tf.expand_dims(test_feed, axis=0)

results = unet.predict(test_feed)

predicted_image = results[0, :, :, 0]  # Extract the 2D image from the batch

fig, axs = plt.subplots(1, 4, figsize=(15, 15))  # Adjust the figsize as needed

axs[0].imshow(fixed_image, cmap='gray')
axs[0].set_title('fixed image')

axs[1].imshow(test_image, cmap = 'gray')
axs[1].set_title('moving image')

axs[2].imshow(predicted_image, cmap= 'gray')
axs[2].set_title('deformed image')

axs[3].imshow(fixed_image,cmap='gray')
axs[3].imshow(predicted_image, alpha = 0.6)
axs[3].set_title('overlapping image')
plt.show()

# obtaining the image directly via the unet
test_image_list = list(test_images_only)
test_image = test_image_list[30]
test_feed = tf.stack([fixed_image, test_image],-1)
test_feed = tf.expand_dims(test_feed, axis=0)


results = unet.predict(test_feed)

predicted_image = results[0, :, :, 0]  # Extract the 2D image from the batch

fig, axs = plt.subplots(1, 4, figsize=(15, 15))  # Adjust the figsize as needed

axs[0].imshow(fixed_image, cmap='gray')
axs[0].set_title('fixed image')

axs[1].imshow(test_image, cmap = 'gray')
axs[1].set_title('moving image')

axs[2].imshow(predicted_image, cmap= 'gray')
axs[2].set_title('deformed image')

axs[3].imshow(fixed_image,cmap='gray')
axs[3].imshow(predicted_image, alpha = 0.6)
axs[3].set_title('overlapping image')
plt.show()