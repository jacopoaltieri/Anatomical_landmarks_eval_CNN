"""
AUTHORS: Altieri J. , Mazzini V.

This program will train two models to perform cephalometric landmark detection.
The first model is a U-Net to do a coarse detection of each landmark ROI;
The second model is a ResNet50 which will perform the actual keypoint detection.

A data augmentation process is also possible and present as a function, beware that this might be time-consuming.
"""

import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
import unet, resnet


batch_size = 20

# ================================================================#
#              Importing dataset from directory                  #
# ================================================================#
input_path = r"C:\Users\jacop\Desktop\DL_Project\processed_dataset"


# =========== Images ===========#
def process_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    img = img / 255  # normalize image
    return img


train_images = tf.data.Dataset.list_files(
    input_path + "/images/train/*.jpg", shuffle=False
)
train_images = train_images.map(process_image)
test_images = tf.data.Dataset.list_files(
    input_path + "/images/test/*.jpg", shuffle=False
)
test_images = test_images.map(process_image)


# =========== Labels ===========#
def load_labels(path):
    with open(path.numpy(), "r", encoding="utf-8") as file:
        lines = file.readlines()[1:]
    labels = [line.strip().split("\t") for line in lines]
    x_values = [float(label[1]) for label in labels]
    y_values = [float(label[2]) for label in labels]
    landmarks = np.concatenate((x_values, y_values))  # [x1,y1,x2,y2...,x14,y14]
    return landmarks


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

# =========== Combine Images and Labels ===========#
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(2000)
train = train.batch(batch_size)
train = train.prefetch(4)   # preload images to avoid bottlenecking

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1000)
test = test.batch(batch_size)
test = test.prefetch(4)   # preload images to avoid bottlenecking
