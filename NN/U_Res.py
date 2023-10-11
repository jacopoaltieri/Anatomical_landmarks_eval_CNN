"""
AUTHORS: Altieri J. , Mazzini V.

This program will train two models to perform cephalometric landmark detection.
The first model is a U-Net to do a coarse detection of each landmark ROI;
The second model is a ResNet50 which will perform the actual keypoint detection.

A data augmentation process is also possible and present as a function, beware that this might be time-consuming.
"""

import os
import sys
import glob
from tqdm import tqdm
import cv2

from tensorflow import keras 
from tensorflow.python.keras.layers import Input,Conv2D, MaxPooling2D,Dropout, LeakyReLU,UpSampling2D, concatenate
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras import Model 

##################################################################
#                        U-Net backbone                          #
##################################################################
ndim = 2
unet_input_features = 2 #(f)ixed and (m)oving image
input_shape = (240,240, unet_input_features)


def unet(pretrained_weights = None, input_size = (240,240, 2,)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = LeakyReLU(alpha=0.01), padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = adam_v2(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(model.shape())

    if(pretrained_weights): model.load_weights(pretrained_weights)
    return model

unet()
# Write here your dataset path
PATH = r"C:\Users\jacop\Desktop\DL_Project\processed_dataset"

images = glob.glob("**/*.jpg", root_dir=PATH, recursive=True)
labels = glob.glob("**/*.txt", root_dir=PATH, recursive=True)

# load images in memory and normalize them
for image in images:
    img = cv2.imread(PATH+'/'+image,0)
    im_normalized = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
