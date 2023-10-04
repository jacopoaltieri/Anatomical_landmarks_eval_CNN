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


input_path=input("Provide the path for the dataset: ")

print("Preprocessing started...")

# Creating the directory tree
print("Creating the directory tree in the current wd...")
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

# finding the matching img-label pairs
lab = set([os.path.splitext(x)[0] for x in label_files])
im = set([os.path.splitext(x)[0] for x in img_files])
matching = list(im.intersection(lab))


# resizing the images and scaling the labels
NEW_SIZE = 240
print("Rescaling images and labels to 240x240 pixels...")
for i, j in enumerate(tqdm(matching)):
    with Image.open(input_path+"/"+j+".jpg").convert('RGB') as im:
        original_size=im.size
        resized_image = im.resize((int(NEW_SIZE), int(NEW_SIZE)))
        resized_image.save(os.getcwd()+"/processed_dataset/images/"+str(i)+".jpg", 'JPEG')

    scale_factor=NEW_SIZE/np.array(original_size)

    path = input_path+"/"+j+".txt"
    data = pd.read_csv(path, sep="[;,\\t]", engine="python")
    coords = np.array(list(zip(data.loc[:, "X"], data.loc[:, "Y"])))
    
    # coords formatting according to YOLO guidelines
    norm_coords = (coords - np.min(coords))/(np.max(coords)-np.min(coords))
    with open(os.getcwd()+"/processed_dataset/labels/"+str(i)+".txt", 'w') as f:
        f.write(f"0,0.5,0.5,1,1,")
        for (x,y) in norm_coords:
            f.write(f"{x},{y},")
print('Preprocessing complete! The dataset in "'+os.getcwd()+'/processed_dataset" is ready to be used!')