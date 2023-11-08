# Identification of anatomical landmarks using CNN

## Introduction
The task of this project is to develop a CNN to perform a coarse detection of 14 anatomical landmarks from lateral cephalometric radiographs.

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
* **preprocessing_scripts**: contains the preprocessor script: use it if you are willing to work with the original dataset. It will find all the matching image-label pairs and sort them into three categories (train, test, val), after resizing the coordinates to the input size of the CNN.

* **NN**: contains the scripts for training and for predictions, the pre-trained model, its training history and the displacement field outputted by the model. It also contains the image-label pair chosen as the fixed image during training.

* **Results**: contains the Loss and Metric plots of the pre-trained model, along with some output examples.
  
## Dataset structure
The main directory needs two subfolders called "images" and "labels" respectively. Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.

## Usage
The scripts run in **python 3.10.12**, please use this version to ensure compatibility, especially if you wish to use the pre-trained model.

#### Requirements
- keras==2.14.0
- matplotlib==3.8.0
- numpy==1.26.1
- opencv_python_headless==4.8.1.78 (or opencv_python==4.8.1.78 if running on Windows)
- pandas==2.1.2
- tensorflow==2.14.0
- tensorflow_addons==0.22.0
- tqdm==4.66.1

First run the following command to ensure that all the required packages are installed and up to date:
* If on Windows: `pip install -r .\requirements_windows.txt`.
* If on Linux: `pip install -r .\requirements_linux.txt`.

### Preprocessing
**If working with the original dataset**, you need to pass it to the preprocessor before feeding it to the NN.
Extract the dataset and download *preprocessing_scripts/dataset_preprocessor.py*. Run the python script, it will create the processed and rescaled dataset in your current working directory. You can choose the minimum size of the images and the train/test/val split percentage by editing the corresponding Parameters in the script.

If you are instead working with your personal dataset, please make sure that the set you feed to the network has the following structure:
Two directories, "images" and "labels", each containing three subfolders named "train", "test" and "val".
The images should be 256x256 pixels in RGB, and the labels should be a tab-separated .txt file containing a header and then 14 rows with the index of the landmark, its X coordinate and its Y coordinate. Corresponding images and labels should have the same name.
An example of the correct formatting for images and labels can be found as the "fixed image/label" in the *NN* directory of this repository.

### Network
This neural network uses a U-Net architecture to perform deformable image registration, which involves aligning two images by deforming elastically one of them to match the other. The network takes in two images - a fixed image and a moving image - and outputs a deformed version of the moving image that aligns with the fixed image. [[1]](#1).
It then deforms the fixed landmark position onto the moving image via the inverse of the displacement field, and plots the moving image with ROIs around the predicted landmark position.

Download the folder *NN*. If you wish to train your own network, use the script "*training.py*"; here you can set the dataset path, the hyperparameters and the name of the saved model. It will output plots for MSE, Loss and an example of deformation.

When you are happy with the results, input the displacement model into the script "*evaluation.py*" and run the model on a full unlabeled dataset. It will output a folder containing the images with the predicted ROIs and .txt files containing their coordinates for each landmark.

## References
<a id="1">[1]</a>Song et al., Automatic cephalometric landmark detection on X-ray images using a deep-learning method