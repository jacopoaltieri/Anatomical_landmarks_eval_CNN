"""
AUTHORS: Altieri J. , Mazzini V.

This module will rearrange the dataset in the correct format in order for it to be fed to the NN; it will also resize all the images (and the coordinates)
The processed dataset will be stored in "processed_dataset" in your current working directory.

You need to run this code only if working from the original dataset
Note that any non-standardized data will have to be processed manually.
"""

import glob
import os
from tqdm import tqdm
from PIL import Image
import numpy as np


# input_path=input("Provide the path for the dataset: ")
input_path = (
    "dataset_teleradiografie_14punti"  # hard-coded for now to speed up the workflow
)

print("Starting the preprocessing...")

# Creating the directory tree
print("creating the directory tree in the current wd...\n")
os.makedirs(os.getcwd() + "/processed_dataset/images", exist_ok=True)
os.makedirs(os.getcwd() + "/processed_dataset/labels", exist_ok=True)


# listing and matching images and labels
print("Finding the image-label pairs...")
for filename in os.listdir(input_path):
    if " " in filename:
        new_filename = filename.replace(" ", "-")
        old_filepath = os.path.join(input_path, filename)
        new_filepath = os.path.join(input_path, new_filename)

label_files = glob.glob("**/*[!README].txt", root_dir=input_path, recursive=True)
img_files = [
    file
    for file in glob.glob("**/*.jpg", root_dir=input_path, recursive=True)
    if "-" not in file
]

lab = set([os.path.splitext(x)[0] for x in label_files])
im = set([os.path.splitext(x)[0] for x in img_files])
matching = list(im.intersection(lab))
print(len(matching))

# resizing the images and scaling the labels
SIZE = 240
original_size=[]
print("rescaling the images...")
for i in tqdm(matching):
    with Image.open(input_path+"/"+i+".jpg").convert('RGB') as im:
        original_size.append(im.size)
        resized_image = im.resize((int(SIZE), int(SIZE)), Image.LANCZOS)
        #resized_image.save(os.getcwd()+"/processed_dataset/images/"+i+".jpg", 'JPEG')

