# Identification of anatomical landmarks using CNN

## Introduction
The task of this project is to develop a CNN to identify 14 anatomical landmarks from lateral cephalometric radiographs.

## Cephalometry principles
To carry out cephalometry, the X-ray tube is placed at 1.5 meters away from the mid sagittal plane, while the detector is at 15 cm on the other side.
In a lateral cephalometric radiograph, the one analyzed by this CNN, the X-ray beam is perpendicular to the patient's sagittal plane in a [Natural head position](https://en.wikipedia.org/wiki/Natural_head_position).

### Cephalometric landmarks
From these images some **cephalometric landmarks** can be extrapolated, which in turn can be joined by lines to form axes, angles and planes, useful to identify cranial deformities or malocllusions.
The 14 landmarks to be found by this network are listed below, ordered according to the dataset labels ([source](https://en.wikipedia.org/wiki/Cephalometric_analysis#Cephalometric_landmarks)):
1. **Na, Nasion**: most anterior point on frontonasal suture
2. **O, Orbitale**: most inferior point on margin of orbit
3. **ANS, Anterior Nasal Spine**: anterior point on maxillary bone
4. **PNS, Posterior Nasal Spine**: posterior limit of bony palate or maxilla
5. **A, Downs A**: most concave point of anterior maxilla
6. **PM, Suprapogonion**: point at which shape of symphysis mentalis changes from convex to concave
7. **Pg, Pogonion**: most anterior point of the mandibular symphysis
8. **Me, Menton**: lowest point on mandibular symphysis
9. **Ba, Basion**: most anterior point on foramen magnum
10. **Pr, Porion**: most superior point of outline of external auditory meatus
11. **S, Sella turcica**: midpoint of sella turcica
12. **CM, Intermolar contact**: point of intermolar contact, i.e. the interocclusal ratio between the first permanent molars
13. **PT, PT point**: point at junction between the lower edge of the foramen rotundum and the posterior edge of the maxillary pterygomaxial fissure.
14. **Go, Gonion**: most posterior inferior point on angle of mandible. Can also be constructed by bisecting the angle formed by intersection of mandibular plane and ramus of mandible

## Repository organization
* **preprocessing_scripts**: contains the preprocessor script: use it if you are willing to work with the original dataset. It will find all the matching image-label pairs and sort them into three categories (train, test, val).
* **NN**: contains the Network, the pre-trained model and its training history. It also contains the image-label pair chosen as the fixed image during training.
* **Outputs**: contains the Loss and Metric plots of the pre-trained model, along with some output examples
  
## Dataset structure
The main directory needs two subfolders called "images" and "labels" respectively. Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.

## Usage
The script runs in python 3.10.12, please use this version to assure compatibility, expecially if you wish to use the pre-trained model.

First run the following command to ensure that all the required packages are installed and up to date: `pip install -r .\requirements.txt`.
### Preprocessing
**If working with the original dataset**, you need to pass it to the preprocessor before feeding it to the NN.
Extract the dataset and download *preprocessing_scripts/dataset_preprocessor.py*. Run the python script, it will create the processed and rescaled dataset in your current working directory. You can choose the minimum size of the images and the train/test/val split percentage by editing the corresponding Parameters in the script.

If you are instead working with your personal dataset, please make sure that the set you feed to the network has the following structure:
Two directories, "images" and "labels", each containing three subfolders named "train", "test" and "val".
The images should be 256x256 px, and the labels should be a tab-separated .txt file containing a header and then 14 rows with the index of the landmark, its X coordinate and its Y coordinate. Corresponding images and labels should have the same name.
An example of the correct formatting for images and labels can be found as the "fixed image/label" in the *NN* directory of this repository.

### Network
Download the script *NN/main.py*. In the first section, set the paths to the processed dataset, the image-label pair chosen as the fixed one (you can use the one provided in the same folder); then choose if the network should be trained or should work on a pre-trained model.

