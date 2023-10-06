"""
AUTHORS: Altieri J. , Mazzini V.

This module will do data augmentation on the "augmented dataset" by flipping the images and their respective coordinates and changing their brightness and contrast.
It will keep the original folder structure, so that both the normal and the augmented dataset can be used in the CNN with the same implementation.
The folder "augmented_dataset" in your current working directory will be created and contain both the original and the augmented images.
"""

import os
import pandas as pd
import glob
import albumentations as alb
import cv2
import shutil
import numpy as np
from tqdm import tqdm

NEW_SIZE = 240  # new image size (chosen to be the minimum of the original sizes)
TRAIN_PERC = 0.75 # percentage of data to put in train set, the other will go in validation

# function to write coords according to YOLO guidelines
def write_coords(coords):
    f.write(f"0 0.5 0.5 1 1 ")
    for (x,y) in coords:
        f.write(f"{x} {y} ")
        
        
input_path=input("Provide the path for the dataset: ")

print("Dataset augmentation started...")
os.makedirs(os.getcwd() + "/augmented_dataset/images/train", exist_ok=True)
os.makedirs(os.getcwd() + "/augmented_dataset/images/test", exist_ok=True)
os.makedirs(os.getcwd() + "/augmented_dataset/labels/train", exist_ok=True)
os.makedirs(os.getcwd() + "/augmented_dataset/labels/test", exist_ok=True)
# transformation pipeline: a flip (hor,vert or both) and a brightness/contrast change
transform = alb.Compose(
    [
        alb.RandomBrightnessContrast(),
        alb.Flip(p=1),
    ],
    keypoint_params=alb.KeypointParams(format="xy"),
)


# get all the images and labels
label_files = glob.glob("**/*.txt", root_dir=input_path, recursive=True)
img_files = glob.glob("**/*.jpg", root_dir=input_path, recursive=True)

# transform the images and save both the original and the augmented
for i, n in enumerate(tqdm(img_files)):
    img = cv2.imread(input_path + "/" + n)
    coord_file = pd.read_csv(
        input_path + "/" + label_files[i],sep=" ", header=None)
    x_columns = coord_file.iloc[:, 5::2].values.tolist()
    y_columns = coord_file.iloc[:, 6::2].values.tolist()
    coordinates = [list(pair) for pair in zip(*x_columns, *y_columns)]
    scaled_coords = np.clip(np.array(coordinates)*NEW_SIZE,0,239)

    transformed = transform(image=img, keypoints=scaled_coords)
    if i<int(len(img_files)*TRAIN_PERC):        
        imgor= os.getcwd() + "/augmented_dataset/images/train/" + str(i) + ".jpg"
        imgaug = os.getcwd() + "/augmented_dataset/images/train/" + str(i) + "aug.jpg"
    else:
        imgor= os.getcwd() + "/augmented_dataset/images/test/" + str(i) + ".jpg"
        imgaug = os.getcwd() + "/augmented_dataset/images/test/" + str(i) + "aug.jpg"   
    cv2.imwrite(imgor, img)
    cv2.imwrite(imgaug, transformed["image"])

    labor = input_path + "/"+label_files[i]
    if i<int(len(img_files)*TRAIN_PERC):        
        labaug = os.getcwd() + "/augmented_dataset/labels/train/" + str(i) + "aug.txt"
        shutil.copy(labor,os.getcwd() + "/augmented_dataset/labels/train/" + str(i) + ".txt")
    else:
        labaug = os.getcwd() + "/augmented_dataset/labels/test/" + str(i) + "aug.txt"
        shutil.copy(labor,os.getcwd() + "/augmented_dataset/labels/test/" + str(i) + ".txt")

        
    norm_coords = np.array(transformed["keypoints"])/NEW_SIZE
    with open(labaug, "w") as f:
        write_coords(norm_coords)
print('Augmentation complete! The dataset in "'+os.getcwd()+'/augmented_dataset" is ready to be used!')