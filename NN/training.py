"""
AUTHORS: Altieri J. , Mazzini V.

This program will train a U-Net to do a coarse detection of cephalometric landmarks.
It creates a "displacement field" which performs an elastic deformation of the "moving image"
onto a fixed one. this also moves the landmarks and their corresponding RoI.

The net architecture is inspired by the one proposed in the following paper:
https://www.jstage.jst.go.jp/article/transinf/E104.D/8/E104.D_2021EDP7001/_pdf
"""

import os
import pickle
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

warnings.filterwarnings("ignore")  # suppress tfa deprecated warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ignore TF unsupported NUMA warnings

from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import tensorflow as tf
import tensorflow_addons as tfa

warnings.resetwarnings()  # restore warnings


# Check for GPU availability
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    print("We got a GPU!")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Sorry, no GPU for you...")

# Avoid OOM errors by setting GPU Memory Consumption Growth
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


######################################## HYPERPARAMETERS AND OTHER OPTIONS ########################################

# Dataset path
input_path = "Anatomical_landmarks_eval_CNN/processed_dataset"
if not os.path.exists(input_path):
    input_path = input("Cannot find the dataset path provided in the script.\n"
                       "Please input a valid path: ")

# Chose the fixed image-label pair you want to train on
fixed_image_path = "fixed_img.jpg"
fixed_label_path = "fixed_lab.txt"

# Choose if the model should be trained
TRAINING = False

# Choose the name of the model to save/load
MODEL_NAME = "unet_pretrained.keras"
TRAINING_HISTORY = "unet_pretrained_history.pickle"

# Training hyperparameters
BATCH_SIZE = 3
EPOCHS = 100
LRELU_ALPHA = 0.01  # alpha coefficient of LeakyReLU
L2REG = 0.0001  # kernel regularizer coefficient

# Loss laplacian penalization coefficient
LAPL_COEF = 0.7

# Adam hyperparameters
LR = 0.0001
BETA_1 = 0.95
BETA_2 = 0.999
EPSILON = 1e-08

# ReduceLROnPlateau hyperparameters
RLR_FACTOR = 0.2
RLR_PATIENCE = 5
RLR_MIN = 0.00001

# EarlyStopping hyperparameters
ES_PATIENCE = 10

print(
    f"Defaulting to the following parameters:\n"
    f"Dataset path: {input_path}\n"
    f"Training mode: {TRAINING}\n"
    f"Model name: {MODEL_NAME}\n"
    f"Training history name: {TRAINING_HISTORY}\n"
    f"Batch size: {BATCH_SIZE}\n"
    f"Epochs: {EPOCHS}\n"
    f"LeakyReLU alpha: {LRELU_ALPHA}\n"
    f"Kernel regularizer coefficient (L2REG): {L2REG}\n"
    f"Learning rate (LR): {LR}\n"
    f"Adam Beta_1: {BETA_1}\n"
    f"Adam Beta_2: {BETA_2}\n"
    f"Adam Epsilon: {EPSILON}\n"
    f"ReduceLROnPlateau factor: {RLR_FACTOR}\n"
    f"ReduceLROnPlateau patience: {RLR_PATIENCE}\n"
    f"ReduceLROnPlateau min LR: {RLR_MIN}\n"
    f"Laplacian penalization coefficient: {LAPL_COEF}\n"
    f"Early Stopping patience: {ES_PATIENCE}\n"
)


######################################## Importing Dataset #######################################################

print("Started dataset loading...")

# =========== Functions =========== #

def process_image(x):
    """Reads the file as a sequence of bytes and converts it
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
            x_value = float(columns[1])
            y_value = float(columns[2])
            landmarks.extend([x_value, y_value])
    return np.array(landmarks)


# For augmentation
def apply_noise(image, max_noise_stddev=0.1):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=max_noise_stddev)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image

def apply_brightness(image, max_delta=0.2):
    delta = tf.random.uniform((), minval=-max_delta, maxval=max_delta)
    image = tf.image.adjust_brightness(image, delta)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def apply_contrast(image, max_contrast_factor=1.5):
    contrast_factor = tf.random.uniform((), minval=1.0, maxval=max_contrast_factor, dtype=tf.float32)
    image = tf.image.adjust_contrast(image, contrast_factor)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def augment_image(x):
    """Reads the file as a sequence of bytes and converts it
    into a grayscale JPG image normalized in [0,1], then augments it

    Args:
        x (str): path to the image

    Returns:
        img: processed image
    """
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.rgb_to_grayscale(img)
    img = img / 255

    img = apply_noise(img)
    img = apply_brightness(img)
    img = apply_contrast(img)
    
    img = tf.squeeze(img, axis=-1)
    return img

# =========== Images =========== #

# Create train, test, val datasets and process the images
train_images = tf.data.Dataset.list_files(
    input_path + "/images/train/*.jpg", shuffle=False
).map(process_image)
test_images = tf.data.Dataset.list_files(
    input_path + "/images/test/*.jpg", shuffle=False
).map(process_image)
val_images = tf.data.Dataset.list_files(
    input_path + "/images/val/*.jpg", shuffle=False
).map(process_image)

# Create the augmented datasets
aug_train_images = tf.data.Dataset.list_files(
    input_path + "/images/train/*.jpg", shuffle=False
).map(augment_image)
aug_test_images = tf.data.Dataset.list_files(
    input_path + "/images/test/*.jpg", shuffle=False
).map(augment_image)
aug_val_images = tf.data.Dataset.list_files(
    input_path + "/images/val/*.jpg", shuffle=False
).map(augment_image)

# Processing the fixed image
fixed_image = tf.data.Dataset.list_files(fixed_image_path, shuffle=False).map(
    process_image
)


# =========== Labels =========== #

# Create train, test and val label dataset and load their labels
train_labels = tf.data.Dataset.list_files(
    input_path + "/labels/train/*.txt", shuffle=False
).map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))
test_labels = tf.data.Dataset.list_files(
    input_path + "/labels/test/*.txt", shuffle=False
).map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))
val_labels = tf.data.Dataset.list_files(
    input_path + "/labels/val/*.txt", shuffle=False
).map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

# Processing the fixed image's labels
fixed_label = tf.data.Dataset.list_files(fixed_label_path, shuffle=False).map(
    lambda x: tf.py_function(load_labels, [x], [tf.float16])
)

# =========== Combine Images and Labels =========== #

# The new dataset has also the augmented images
train_dataset = tf.data.Dataset.zip((train_images, train_labels)).concatenate(tf.data.Dataset.zip((aug_train_images, train_labels))).shuffle(1000)
test_dataset = tf.data.Dataset.zip((train_images, train_labels)).concatenate(tf.data.Dataset.zip((aug_test_images, test_labels))).shuffle(1000)
val_dataset = tf.data.Dataset.zip((train_images, train_labels)).concatenate(tf.data.Dataset.zip((aug_val_images, val_labels))).shuffle(1000)

fixed_dataset = tf.data.Dataset.zip((fixed_image, fixed_label))


######################################## U-NET #######################################################

# =========== Generators =========== #
print("Creating generators...")

# Extract only the images
train_images_only = train_dataset.map(lambda img, lbl: img)
test_images_only = test_dataset.map(lambda img, lbl: img)
val_images_only = val_dataset.map(lambda img, lbl: img)
fixed_image_only = fixed_dataset.map(lambda img, lbl: img)

# Make a list containing the images
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
        combined_image = tf.stack([fixed_image, moving_image], -1)
        yield (combined_image, fixed_image)


# Create a TensorFlow dataset from the generator
train_images_dataset = tf.data.Dataset.from_generator(
    train_image_generator,
    output_signature=(
        tf.TensorSpec(shape=(256, 256, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256), dtype=tf.float32),
    ),
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
        tf.TensorSpec(shape=(256, 256), dtype=tf.float32),
    ),
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
        tf.TensorSpec(shape=(256, 256), dtype=tf.float32),
    ),
)
test_images_dataset = test_images_dataset.batch(BATCH_SIZE)


# =========== Network Architecture =========== #
# a: activation, c: convolution,
# p: pooling, u: upconvolution, bn: batch normalization


input_shape = (256, 256, 2)  # Two images, one fixed and one moving
input = tf.keras.layers.Input(shape=input_shape)

### Encoder ###

a1 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(input)
c1 = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2REG))(a1)
bn1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1)

a2 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(p1)
c2 = tf.keras.layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2REG))(a2)
bn2 = tf.keras.layers.BatchNormalization()(c2)
c2 = tf.keras.layers.Dropout(0.1)(c2)
p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c2)

a3 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(p2)
c3 = tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2REG),)(a3)
bn3 = tf.keras.layers.BatchNormalization()(c3)
c3 = tf.keras.layers.Dropout(0.2)(c3)
p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c3)

a4 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(p3)
c4 = tf.keras.layers.Conv2D(512, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2REG))(a4)
bn4 = tf.keras.layers.BatchNormalization()(c4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

### Bottleneck ###

a5 = tf.keras.layers.LeakyReLU(alpha=LRELU_ALPHA)(p4)
c5 = tf.keras.layers.Conv2D(1024, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2REG))(a5)
bn5 = tf.keras.layers.BatchNormalization()(c5)
c5 = tf.keras.layers.Dropout(0.3)(c5)

### Decoder ###

u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
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

# Create the displacement field
displacement_field = tf.keras.layers.Conv2D(
    2, kernel_size=3, activation="linear", padding="same", name="disp"
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
    Apply a deformation transformation to an image using a displacement field.

    Args:
        inputs (list): A list containing two TensorFlow tensors, 'image' and 'displacement_field'.

    Returns:
        tf.Tensor: The deformed image resulting from applying the displacement field to the input image.
    """
    image, displacement_field = inputs
    deformed_image = tfa.image.dense_image_warp(image, -displacement_field)
    return deformed_image


def_image = tf.keras.layers.Lambda(apply_deformation)(
    [moving_image, displacement_field]
)
output = def_image


# =========== Model Initialization =========== #
unet = tf.keras.Model(inputs=input, outputs=output)

def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    laplacian = tf.reduce_mean(tf.abs(tf.image.image_gradients(tf.expand_dims(y_pred, axis=-1))))
    total_loss = mse + LAPL_COEF * laplacian
    return total_loss

# Define the callback to reduce the learning rate when the validation loss stops improving
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=RLR_FACTOR,
    patience=RLR_PATIENCE,
    min_lr=RLR_MIN,
    amsgrad=False,
)

# Implement Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=ES_PATIENCE, verbose=1, mode='min', restore_best_weights=True)
callbacks = [reduce_lr, early_stopping]

adam = tf.keras.optimizers.Adam(
    learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON
)

unet.compile(optimizer=adam, loss="mse", metrics="mse")


# =========== Model Training =========== #
if TRAINING:
    print("Training the model...")
    history = unet.fit(
        train_images_dataset,
        epochs=EPOCHS,
        validation_data=val_images_dataset,
        callbacks=callbacks,
    )

    # Save the model
    model_name = MODEL_NAME
    training_hist_name = TRAINING_HISTORY
    unet.save(model_name)
    print("Model saved as", model_name)
    # Save the history via pickle
    with open(training_hist_name, "wb") as file_pi:
        pickle.dump(history.history, file_pi)


# Load a pretrained model
unet = tf.keras.models.load_model(MODEL_NAME, safe_mode=False, custom_objects={"custom_loss": custom_loss})
# Load the training history
with open(TRAINING_HISTORY, "rb") as file_pi:
    history = pickle.load(file_pi)

# Plot training & validation metric values
plt.plot(history["mse"])
plt.plot(history["validation_mse"])
plt.title("Mean Squared Error")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.savefig("mse.png")
plt.close()
print(f"Plot saved as {os.getcwd()}'/mse.png'")

# Plot training & validation loss values
plt.plot(history["loss"])
plt.plot(history["validation_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.savefig("loss.png")
plt.close()
print(f"Plot saved as {os.getcwd()}'/loss.png'")


######################################## RESULTS #######################################################

# =========== Model Testing =========== #
print("Evaluating on test dataset...")
results = unet.evaluate(test_images_dataset)
print("Test loss:", results[0])
print("Test accuracy:", results[1])


# =========== Extracting an example to plot =========== #
example = test_dataset.as_numpy_iterator().next()
test_image = example[0]
test_labels = example[1][0]

# Stack tensors to obtain the correct input for the U-net
test_feed = tf.stack([fixed_image, test_image], -1)
test_feed = tf.expand_dims(test_feed, axis=0)

# Creating the model that outputs the displacement field
displacement_model = tf.keras.Model(
    inputs=unet.input, outputs=unet.get_layer("disp").output
)
displacement_field = displacement_model.predict(test_feed)

# Expand the images to match displacement_field dimensions
test_image_expanded = np.expand_dims(test_image, axis=0)
test_image_expanded = np.expand_dims(test_image_expanded, axis=3)
fixed_image_expanded = np.expand_dims(fixed_image, axis=0)
fixed_image_expanded = np.expand_dims(fixed_image_expanded, axis=3)


# =========== Direct transformation (Moving to Fixed) =========== #

moving_to_fixed = tfa.image.dense_image_warp(test_image_expanded, -displacement_field)

# =========== Inverse transformation (Fixed to Moving) =========== #

inverse_transform = -displacement_field
fixed_to_moving = tfa.image.dense_image_warp(fixed_image_expanded, -inverse_transform)


# =========== Adding Landmarks and ROI =========== #
fixed_labels = fixed_dataset.as_numpy_iterator().next()[1][0]

def plot_with_landmarks_and_ROI(ax, image, deformed_landmarks, true_landmarks, bbox_size=15):
    # Plot the image on the given axis
    ax.imshow(image, cmap="gray")

    # Create a bounding box around the deformed landmarks in red
    for i in range(0, len(deformed_landmarks), 2):
        x, y = deformed_landmarks[i], deformed_landmarks[i + 1]

        bounding_box = Rectangle(
            (x - bbox_size / 2, y - bbox_size / 2),
            bbox_size,
            bbox_size,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(bounding_box)

    # Plot true landmarks in green
    for i in range(0, len(true_landmarks), 2):
        x, y = true_landmarks[i], true_landmarks[i + 1]
        ax.plot(x, y, "go", markersize=2)

    # Set axis limits to ensure landmarks and bounding boxes are visible
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)



def deform_landmarks(landmarks, displacement_field):
    """
    Deform landmarks based on a displacement field.

    Args:
        landmarks (np.ndarray): An array of shape (N, 2) containing landmark coordinates.
        displacement_field (np.ndarray): The displacement field with shape (1, H, W, 2).

    Returns:
        np.ndarray: Deformed landmarks based on the displacement field.
    """
    deformed_landmarks = []

    for i in range(0, len(landmarks), 2):
        x, y = landmarks[i], landmarks[i + 1]

        # Get the displacement values from the displacement field
        disp_x = displacement_field[0, int(x), int(y), 0]
        disp_y = displacement_field[0, int(x), int(y), 1]

        # Apply displacement to the landmarks
        new_x = x + disp_x
        new_y = y + disp_y

        deformed_landmarks.extend([new_x, new_y])
    return np.array(deformed_landmarks)

# Deform the landmarks of the fixed img onto the moving one
deformed_landmarks = deform_landmarks(fixed_labels, inverse_transform)


# =========== Plotting the results =========== #


fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Image Transformations")

# Plot Fixed Image
axs[0, 0].imshow(fixed_image, cmap="gray")
axs[0, 0].set_title("Fixed Image")

# Plot Moving Image
axs[0, 1].imshow(test_image, cmap="gray")
axs[0, 1].set_title("Moving Image")

# Plot Moving Image overlapped with Fixed Image
axs[0, 2].imshow(fixed_image, cmap="gray")
axs[0, 2].imshow(test_image, alpha=0.4)
axs[0, 2].set_title("Moving Overlapping with Fixed")

# Plot Moving warped to Fixed
axs[0, 3].imshow(moving_to_fixed[0], cmap="gray")
axs[0, 3].set_title("Deformation of Moving")

# Plot Deformed Moving Image overlapped with Fixed Image
axs[1, 0].imshow(fixed_image, cmap="gray")
axs[1, 0].imshow(moving_to_fixed[0], alpha=0.4)
axs[1, 0].set_title("Deformed Moving Overlapping with Fixed")

# Plot Fixed warped to Moving
axs[1, 1].imshow(fixed_to_moving[0], cmap="gray")
axs[1, 1].set_title("Deformation of Fixed")

# Plot Deformed Fixed Image overlapped with Moving Image
axs[1, 2].imshow(test_image, cmap="gray")
axs[1, 2].imshow(fixed_to_moving[0], alpha=0.4)
axs[1, 2].set_title("Deformed Fixed overlapped with Moving")

# Plot Moving Image with Landmarks and ROI
plot_with_landmarks_and_ROI(axs[1, 3], test_image, deformed_landmarks, test_labels)
axs[1, 3].set_title("Moving with predicted Landmarks ROI")

# Remove axis labels and ticks
for ax in axs.flat:
    ax.label_outer()

plt.tight_layout()
plt.savefig("results.png")
plt.close()