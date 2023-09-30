# Identification of anatomical landmarks using CNN

## Introduction
The task of this project is to develop a CNN to identify 14 anatomical landmarks from lateral cephalometric radiographs ([dataset link](https://drive.google.com/drive/folders/1c6Sp1uKYlwv4TCbHMv0JkLjFkdfkNPPB?usp=sharing)).

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

## Usage


## Useful links
* (Original dataset)[https://drive.google.com/drive/folders/1c6Sp1uKYlwv4TCbHMv0JkLjFkdfkNPPB?usp=sharing]
