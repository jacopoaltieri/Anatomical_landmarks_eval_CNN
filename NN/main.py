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


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BATCH_SIZE = 4

# ================================================================ #
#              Importing dataset from directory                    #
# ================================================================ #
#input_path = "/mnt/c/Users/jacop/Desktop/DL_Project/processed_dataset/" #if in wsl
input_path = r"C:\Users\jacop\Desktop\DL_Project\processed_dataset"  # if in windows


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
    with open(path.numpy(), "r", encoding="utf-7") as file:
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


"""
# =========== Show some examples =========== #

data_samples = train.as_numpy_iterator()
res = data_samples.next()

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
"""



# ================================================================ #
#                        U-Net backbone                            #
# ================================================================ #
# a: activation, c: convolution, p: pooling, u: upconvolution

unet_input_features = 2  # fixed and moving image
input_shape = (256, 256, unet_input_features)


inputs = tf.keras.layers.Input(input_shape)    

### Downsampling path ###
a1 = tf.keras.layers.LeakyReLU(alpha=0.01)(inputs)
c1 = tf.keras.layers.Conv2D(64, 3, padding = "same", kernel_initializer = "he_normal")(a1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1)

a2 = tf.keras.layers.LeakyReLU(alpha=0.01)(p1)
c2 = tf.keras.layers.Conv2D(128, 3, padding = "same", kernel_initializer = "he_normal")(a2)
c2 = tf.keras.layers.Dropout(0.1)(c2)
p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c2)

a3 = tf.keras.layers.LeakyReLU(alpha=0.01)(p2)
c3 = tf.keras.layers.Conv2D(256, 3, padding = "same", kernel_initializer = "he_normal")(a3)
c3 = tf.keras.layers.Dropout(0.2)(c3)
p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c3)

a4 = tf.keras.layers.LeakyReLU(alpha=0.01)(p3)
c4 = tf.keras.layers.Conv2D(512, 3, padding = "same", kernel_initializer = "he_normal")(a4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

a5 = tf.keras.layers.LeakyReLU(alpha=0.01)(p4)
c5 = tf.keras.layers.Conv2D(1024, 3, padding = "same", kernel_initializer = "he_normal")(a5)
c5 = tf.keras.layers.Dropout(0.3)(c5)

### Upsampling Path ###
u6 = tf.keras.layers.Conv2DTranspose(1024,(2,2),strides=(2,2),padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
a6 = tf.keras.layers.LeakyReLU(alpha=0.01)(u6)
c6 = tf.keras.layers.Conv2D(512, 3, padding = "same", kernel_initializer = "he_normal")(a6)
c6 = tf.keras.layers.Dropout(0.3)(c6)

u7 = tf.keras.layers.Conv2DTranspose(512,(2,2),strides=(2,2),padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
a7 = tf.keras.layers.LeakyReLU(alpha=0.01)(u7)
c7 = tf.keras.layers.Conv2D(256, 3, padding = "same", kernel_initializer = "he_normal")(a7)
c7 = tf.keras.layers.Dropout(0.2)(c7)

u8 = tf.keras.layers.Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
a8 = tf.keras.layers.LeakyReLU(alpha=0.01)(u8)
c8 = tf.keras.layers.Conv2D(128, 3, padding = "same", kernel_initializer = "he_normal")(a8)
c8 = tf.keras.layers.Dropout(0.2)(c8)

u9 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1])
a9 = tf.keras.layers.LeakyReLU(alpha=0.01)(u9)
c9 = tf.keras.layers.Conv2D(64, 3, padding = "same", kernel_initializer = "he_normal")(a9)
c9 = tf.keras.layers.Dropout(0.1)(c9)

outputs = tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(c9)

unet = tf.keras.Model(inputs=[inputs], outputs=[outputs])

displacement_tensor = tf.keras.layers.Conv2D(2,kernel_size=3, padding='same', name='disp')(unet.output)
# check tensor shape
print('displacement tensor:', displacement_tensor.shape)