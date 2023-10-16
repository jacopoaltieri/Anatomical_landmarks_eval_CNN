"""
AUTHORS: Altieri J. , Mazzini V.

This program will train two models to perform cephalometric landmark detection.
The first model is a U-Net to do a coarse detection of each landmark ROI;
The second model is a ResNet50 which will perform the actual keypoint detection.

A data augmentation process is also possible and present as a function, beware that this might be time-consuming.
"""
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore TF unsupported NUMA warnings
import tensorflow as tf

# import unet, resnet


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BATCH_SIZE = 4

# ================================================================ #
#              Importing dataset from directory                    #
# ================================================================ #
input_path = "/mnt/c/Users/jacop/Desktop/DL_Project/processed_dataset/" #if in wsl
#input_path = r"C:\Users\jacop\Desktop\DL_Project\processed_dataset"  # if in windows


# =========== Images =========== #
def process_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    img = img / 255  # normalize pixel value
    return img


train_images = tf.data.Dataset.list_files(
    input_path + "/images/train/*.jpg", shuffle=False
)
train_images = train_images.map(process_image)
test_images = tf.data.Dataset.list_files(
    input_path + "/images/test/*.jpg", shuffle=False
)
test_images = test_images.map(process_image)
val_images = tf.data.Dataset.list_files(input_path + "/images/val/*.jpg", shuffle=False)
val_images = val_images.map(process_image)


# =========== Labels =========== #
def load_labels(path):
    landmarks =[]
    with open(path.numpy(), "r", encoding="utf-8") as file:
        next(file, None)
        for line in file:
            # Split each line into columns
            columns = line.strip().split('\t')

            # Extract X and Y values and convert them to floats
            x_value = float(columns[1])/256 # normaliza image size
            y_value = float(columns[2])/256
            landmarks.extend([x_value, y_value])
    return np.array(landmarks)


train_labels = tf.data.Dataset.list_files(
    input_path + "/labels/train/*.txt", shuffle=False
)
train_labels = train_labels.map(
    lambda x: tf.py_function(load_labels, [x], [tf.float16])
)
test_labels = tf.data.Dataset.list_files(
    input_path + "/labels/test/*.txt", shuffle=False
)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))
val_labels = tf.data.Dataset.list_files(input_path + "/labels/val/*.txt", shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))


# =========== Combine Images and Labels =========== #
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(1500)
train = train.batch(BATCH_SIZE)
train = train.prefetch(4)  # preload images to avoid bottlenecking

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1000)
test = test.batch(BATCH_SIZE)
test = test.prefetch(4)  # preload images to avoid bottlenecking

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(BATCH_SIZE)
val = val.prefetch(4)  # preload images to avoid bottlenecking



# =========== Show some examples =========== #

data_samples = train.as_numpy_iterator()
res = data_samples.next()
print(res)

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    sample_image = res[0][idx]
    sample_coords = res[1][0][idx]
    
    for i in range(0, len(sample_coords), 2):
        x, y = sample_coords[i], sample_coords[i + 1]
        x_pixel = int(x * 256)
        y_pixel = int(y * 256)
        cv2.circle(sample_image, (x_pixel, y_pixel), 2, (255, 0, 0), -1)
    
    ax[idx].imshow(sample_image)
plt.show()