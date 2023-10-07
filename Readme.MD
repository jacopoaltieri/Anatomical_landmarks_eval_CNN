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
* **preprocessing_scripts**: contains the preprocessor and the data augmentation scripts. Use them if you are willing to work with the original dataset or the non-augmented one. If you instead use the "final" dataset provided by us (third link at the bottom of the page), you can ignore this folder.
  
## Dataset structure
The dataset has to have a precise structure both in folder organization and keypoints labeling, in order to be correctly read by the YOLO net
Note that both "processed" and "augmented" dataset already have the correct structure, but we will explain it anyway to allow everybody to change the processing pipeline independently.

The main directory needs two subfolders called "images" and "labels" respectively. Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.

The labeling system is the [ultralytics YOLO format](https://docs.ultralytics.com/datasets/pose/):
* Object class index: 0 (since we are not doing detection);
* Object center coordinates: The x and y coordinates of the center of the object, normalized to be between 0 and 1. In our case both 0.5;
* Object width and height: The width and height of the object, normalized to be between 0 and 1. In our case both 1;
* Object keypoint coordinates: The keypoints of the object, normalized to be between 0 and 1. In our case 14 xy pairs.
Here is an example of the label format for pose estimation task:

`0 0.5 0.5 1 1 px1 py1 px2 py2 ... px14 py14`
## Usage
First run the following command to ensure that all the required packages are installed and up to date: `pip install -r .\requirements.txt`.
### Preprocessing
Based on which dataset you decide to use, you may need some preliminary work:
* **If working with the original dataset**: Extract the dataset and download *preprocessing_scripts/dataset_preprocessor.py*. Run the python script, it will create the processed and rescaled dataset in your current working directory, ready to be fed to the CNN.
  
* **If you wish to do dataset augmentation**: Extract the **processed** dataset and download *preprocessing_scripts/dataset_augment.py*. Run the python script, it will create the augmented dataset in your current working directory, ready to be fed to the CNN.

Note that both the "processed" and the "augmented" dataset have the same folder structure, so they can both be analyzed by the network.

### Network
## Useful links
* [Original dataset](https://drive.google.com/file/d/1c6zAOMGOXeI8-PzsOXNV0kaRDOX9jGsg/view?usp=sharing)
* [Processed dataset](https://drive.google.com/file/d/1b5PBYAQm500APmUFrpmd1U2fENg3asXZ/view?usp=sharing)
* [Augmented dataset](https://drive.google.com/file/d/1c7EFdm-p5a3cv4611V5BElwjzm9YnOeC/view?usp=sharing)
