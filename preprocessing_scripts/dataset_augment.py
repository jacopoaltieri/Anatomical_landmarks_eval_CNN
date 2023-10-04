"""
AUTHORS: Altieri J. , Mazzini V.

This module will do data augmentation on the "processed dataset" by flipping the images and their respective coordinates and changing their brightness and contrast.
It will keep the original folder structure, so that both the normal and the augmented dataset can be used in the CNN with the same implementation.
The folder "augmented_dataset" in your current working directory will be created and contain both the original and the augmented dataset.
"""

import os
import pandas as pd
import glob
import albumentations as alb
import cv2
import shutil
import numpy as np
from tqdm import tqdm

NEW_SIZE = 240

input_path=input("Provide the path for the dataset: ")

print("Dataset augmentation started...")
os.makedirs(os.getcwd()+"/augmented_dataset/images/", exist_ok=True)
os.makedirs(os.getcwd()+"/augmented_dataset/labels/", exist_ok=True)

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
        input_path + "/" + label_files[i], header=None)
    x_columns = coord_file.iloc[:, 5::2].values.tolist()
    y_columns = coord_file.iloc[:, 6::2].values.tolist()
    coordinates = [list(pair) for pair in zip(*x_columns, *y_columns)]
    scaled_coords = np.clip(np.array(coordinates)*NEW_SIZE,0,239)

    transformed = transform(image=img, keypoints=scaled_coords)
    imgor= os.getcwd() + "/augmented_dataset/images/" + str(i) + ".jpg"
    imgaug = os.getcwd() + "/augmented_dataset/images/" + str(i) + "aug.jpg"
    cv2.imwrite(imgor, img)
    cv2.imwrite(imgaug, transformed["image"])

    labor = input_path + "/"+label_files[i]
    labaug = os.getcwd() + "/augmented_dataset/labels/" + str(i) + "aug.txt"
    shutil.copy(labor,os.getcwd() + "/augmented_dataset/labels/" + str(i) + ".txt")

    norm_coords = np.array(transformed["keypoints"])/NEW_SIZE
    with open(labaug, "w") as f:
        f.write(f"0,0.5,0.5,1,1,")
        for (x,y) in norm_coords:
            f.write(f"{x},{y},")