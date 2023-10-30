"""
AUTHORS: Altieri J. , Mazzini V.

This module will rearrange the dataset in the correct format in order for it to be fed to the NN;
The processed dataset will be stored in "processed_dataset" in your current working directory.
It will find the matching image-labels pair from the original dataset and sort them into train, test and val folders based on a user-chosen percentage
It will also automatically resize images and labels according to a user-chosen size, and discard all the images smaller than a threshold

You need to run this code only if working from the original dataset.
"""

import glob
import os
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd


###### Functions ######

def write_labels(path,coords):
    with open(path, 'w') as f:
        f.write("Landmark\tX\tY\n")
        for i, (x,y) in enumerate(coords, start=1):
            f.write(f"{i}\t{x}\t{y}\n")

def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)


###### Parameters ######

MIN_SIZE = 300 # minimum required size

# percentages of train,val and test dataset
TRAIN_PERC = 0.60 
VAL_PERC = 0.20
TEST_PERC = 0.20


input_path=input("Provide the path for the dataset: ")

print("Preprocessing started...")

# Creating the directory tree
print(f"Creating the directory tree in{os.getcwd()}")
os.makedirs(os.getcwd() + "/processed_dataset/images/train", exist_ok=True)
os.makedirs(os.getcwd() + "/processed_dataset/images/test", exist_ok=True)
os.makedirs(os.getcwd() + "/processed_dataset/images/val", exist_ok=True)
os.makedirs(os.getcwd() + "/processed_dataset/labels/train", exist_ok=True)
os.makedirs(os.getcwd() + "/processed_dataset/labels/test", exist_ok=True)
os.makedirs(os.getcwd() + "/processed_dataset/labels/val", exist_ok=True)


# listing and matching images and labels
print("Finding the image-label pairs...")
for filename in os.listdir(input_path):
    if " " in filename:
        new_filename = filename.replace(" ", "-")
        old_filepath = os.path.join(input_path, filename)
        new_filepath = os.path.join(input_path, new_filename)


    
label_files = glob.glob("**/*[!README].txt", root_dir=input_path, recursive=True)
label_files = [file for file in label_files if count_lines(input_path+"/"+file) == 15]  #removes files with more/less than 14 landmarks

img_files = [file for file in glob.glob("**/*.jpg", root_dir=input_path, recursive=True) if "-" not in file]

    
# finding the matching img-label pairs
lab = set([os.path.splitext(x)[0] for x in label_files])
im = set([os.path.splitext(x)[0] for x in img_files])
matching = list(im.intersection(lab))

# resizing the images and scaling the labels
print("Rescaling images to 256x256 pixels...")
for i, j in enumerate(tqdm(matching)):
    im = cv2.imread(input_path+"/"+j+".jpg", cv2.IMREAD_UNCHANGED)
    original_size = im.shape[:2]
    if max(original_size) < MIN_SIZE:
        continue  # Discard images smaller than threshold
        
    # converting grayscale images to rgb
    if len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    
    resized_image = cv2.resize(im,(int(256),int(256)))

    path = input_path+"/"+j+".txt"
    data = pd.read_csv(path, sep="[;,\\t]", engine="python")
    coords = np.array(list(zip(data.loc[:, "X"], data.loc[:, "Y"])))

    # rescale coords to 256
    scale_x = 256 / original_size[1]  # Calculate scaling factor for X
    scale_y = 256 / original_size[0]  # Calculate scaling factor for Y
    new_coords = coords * np.array([scale_x, scale_y])

    if i<int(len(matching)*TRAIN_PERC):        
        cv2.imwrite(os.getcwd()+"/processed_dataset/images/train/"+str(i)+".jpg", resized_image)
    elif i < int(len(matching) * (TRAIN_PERC + VAL_PERC)):
        cv2.imwrite(os.getcwd()+"/processed_dataset/images/val/"+str(i)+".jpg", resized_image)
    else:
        cv2.imwrite(os.getcwd()+"/processed_dataset/images/test/"+str(i)+".jpg", resized_image)


    if i<int(len(matching)*TRAIN_PERC):        
        write_labels(os.getcwd()+"/processed_dataset/labels/train/"+str(i)+".txt", new_coords) 
    elif i < int(len(matching) * (TRAIN_PERC + VAL_PERC)):
        write_labels(os.getcwd()+"/processed_dataset/labels/val/"+str(i)+".txt", new_coords) 
    else:
        write_labels(os.getcwd()+"/processed_dataset/labels/test/"+str(i)+".txt", new_coords)
        
print('Preprocessing complete! The dataset in "'+os.getcwd()+'/processed_dataset" is ready to be used!')