"""
AUTHORS: Altieri J. , Mazzini V.

This module will rearrange the dataset in the correct format in order for it to be fed to the NN;
The processed dataset will be stored in "processed_dataset" in your current working directory.
It will find the matching image-labels pair from the original dataset and sort them into train, test and val folders based on a user-chosen percentage
It will also automatically resize images and labels according to a user-chosen size

You need to run this code only if working from the original dataset
"""

import glob
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd

###### Functions #####
def label_writer(path,coords):
    with open(path, 'w') as f:
        f.write("Landmark\tX\tY\n")
        for i, (x,y) in enumerate(coords, start=1):
            f.write(f"{i}\t{x}\t{y}\n")

NEW_SIZE = 256  # new image size
# percentages of train,val and test dataset
TRAIN_PERC = 0.60 
VAL_PERC = 0.20
TEST_PERC = 0.20

input_path=input("Provide the path for the dataset: ")

print("Preprocessing started...")

# Creating the directory tree
print("Creating the directory tree in the current wd...")
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

# Define a function to count lines in a file
def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)
    
label_files = glob.glob("**/*[!README].txt", root_dir=input_path, recursive=True)
label_files = [file for file in label_files if count_lines(input_path+"/"+file) == 15]

img_files = [
    file
    for file in glob.glob("**/*.jpg", root_dir=input_path, recursive=True)
    if "-" not in file
]


    
# finding the matching img-label pairs
lab = set([os.path.splitext(x)[0] for x in label_files])
im = set([os.path.splitext(x)[0] for x in img_files])
matching = list(im.intersection(lab))
print(f"Found {len(matching)} image-label pairs")

# resizing the images and scaling the labels
print("Rescaling images to 256x256 pixels...")
for i, j in enumerate(tqdm(matching)):
    with Image.open(input_path+"/"+j+".jpg").convert('RGB') as im:
        original_size = im.size
        resized_image = im.resize((int(NEW_SIZE),int(NEW_SIZE)))
        if i<int(len(matching)*TRAIN_PERC):        
            resized_image.save(os.getcwd()+"/processed_dataset/images/train/"+str(i)+".jpg", 'JPEG')
        elif i < int(len(matching) * (TRAIN_PERC + VAL_PERC)):
            resized_image.save(os.getcwd()+"/processed_dataset/images/val/"+str(i)+".jpg", 'JPEG')
        else:
             resized_image.save(os.getcwd()+"/processed_dataset/images/test/"+str(i)+".jpg", 'JPEG')
    
    scale_factor = NEW_SIZE/np.array(original_size)
           
    path = input_path+"/"+j+".txt"
    data = pd.read_csv(path, sep="[;,\\t]", engine="python")
    coords = np.array(list(zip(data.loc[:, "X"], data.loc[:, "Y"])))
    coords*= scale_factor
    
    if i<int(len(matching)*TRAIN_PERC):        
        label_writer(os.getcwd()+"/processed_dataset/labels/train/"+str(i)+".txt", coords) 
    elif i < int(len(matching) * (TRAIN_PERC + VAL_PERC)):
        label_writer(os.getcwd()+"/processed_dataset/labels/val/"+str(i)+".txt", coords) 
    else:
        label_writer(os.getcwd()+"/processed_dataset/labels/test/"+str(i)+".txt", coords) 
print('Preprocessing complete! The dataset in "'+os.getcwd()+'/processed_dataset" is ready to be used!')