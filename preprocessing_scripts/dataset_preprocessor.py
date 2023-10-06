"""
AUTHORS: Altieri J. , Mazzini V.

This module will rearrange the dataset in the correct format in order for it to be fed to the NN;
it will also resize all the images (and their corresponding coordinates)
The processed dataset will be stored in "processed_dataset" in your current working directory.

You need to run this code only if working from the original dataset
"""

import glob
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd


NEW_SIZE = 240  # new image size (chosen to be the minimum of the original sizes)
TRAIN_PERC = 0.75 # percentage of data to put in train set, the other will go in validation

# function to write coords according to YOLO guidelines
def write_coords(coords):
    f.write(f"0 0.5 0.5 1 1 ")
    for (x,y) in coords:
        f.write(f"{x} {y} ")


input_path=input("Provide the path for the dataset: ")

print("Preprocessing started...")

# Creating the directory tree
print("Creating the directory tree in the current wd...")
os.makedirs(os.getcwd() + "/processed_dataset/images/train", exist_ok=True)
os.makedirs(os.getcwd() + "/processed_dataset/images/test", exist_ok=True)
os.makedirs(os.getcwd() + "/processed_dataset/labels/train", exist_ok=True)
os.makedirs(os.getcwd() + "/processed_dataset/labels/test", exist_ok=True)


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

# finding the matching img-label pairs
lab = set([os.path.splitext(x)[0] for x in label_files])
im = set([os.path.splitext(x)[0] for x in img_files])
matching = list(im.intersection(lab))


# resizing the images and scaling the labels
print("Rescaling images to 240x240 pixels...")
for i, j in enumerate(tqdm(matching)):
    with Image.open(input_path+"/"+j+".jpg").convert('RGB') as im:
        original_size=im.size
        resized_image = im.resize((int(NEW_SIZE), int(NEW_SIZE)))
        if i<int(len(matching)*TRAIN_PERC):        
            resized_image.save(os.getcwd()+"/processed_dataset/images/train/"+str(i)+".jpg", 'JPEG')
        else:
             resized_image.save(os.getcwd()+"/processed_dataset/images/test/"+str(i)+".jpg", 'JPEG')
           
    #scale_factor=NEW_SIZE/np.array(original_size)  # no need to rescale for the YOLO labelling format

    path = input_path+"/"+j+".txt"
    data = pd.read_csv(path, sep="[;,\\t]", engine="python")
    coords = np.array(list(zip(data.loc[:, "X"], data.loc[:, "Y"])))
    norm_coords = (coords - np.min(coords))/(np.max(coords)-np.min(coords))

    if i<int(len(matching)*TRAIN_PERC):        
        with open(os.getcwd()+"/processed_dataset/labels/train/"+str(i)+".txt", 'w') as f:
            write_coords(norm_coords)
    else:
        with open(os.getcwd()+"/processed_dataset/labels/test/"+str(i)+".txt", 'w') as f:
            write_coords(norm_coords)
print('Preprocessing complete! The dataset in "'+os.getcwd()+'/processed_dataset" is ready to be used!')