"""
AUTHORS: Altieri J. , Mazzini V.

This code uses the displacement field given by the training.py script and applies it to 
the whole set of images, saving the images in the "images" folder, with the plotted RoIs, 
and the corresponding labels in the "labels" folder. The txt files will contain the coordinates 
of the bounding boxes, given as "x, y, bbox_size" for each landmark. 
"""


import glob
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from tqdm import tqdm


warnings.filterwarnings("ignore")  # suppress tfa deprecated warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ignore TF unsupported NUMA warnings

import tensorflow as tf
import tensorflow_addons as tfa

warnings.resetwarnings()  # restore warnings


######################################## FUNCTIONS ########################################


def process_image(path):
    """
    Reads the file as a sequence of bytes and converts it
    into a grayscale JPG image normalized in [0,1]

    Args:
        x (str): path to the image

    Returns:
        img: processed image
    """
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (256, 256))
    img = tf.image.rgb_to_grayscale(img)
    img = tf.squeeze(img, axis=-1)
    img = img / 255
    return img


def load_labels(path):
    """
    Load landmark labels from a specified file and return them as a NumPy array.

    Args:
        path (str): The file path to the label data.

    Returns:
        numpy.ndarray: An array containing normalized landmark coordinates.
    """
    landmarks = []
    with open(path, "r", encoding="utf-8") as file:
        next(file, None)
        for line in file:
            columns = line.strip().split("\t")
            x_value = float(columns[1])
            y_value = float(columns[2])
            landmarks.extend([x_value, y_value])
    return np.array(landmarks)


def net_feeder(fixed_image, moving_image):
    """
    Create a TensorFlow tensor to feed two images into the neural network for processing.

    Args:
    - fixed_image (tf.Tensor): The fixed image to be processed.
    - moving_image (tf.Tensor): The moving image to be processed.

    Returns:
    - net_feed (tf.Tensor): A tensor containing both the fixed and moving images with an
                            added batch dimension.
    """
    net_feed = tf.stack([fixed_image, moving_image], -1)
    net_feed = tf.expand_dims(net_feed, axis=0)
    return net_feed


def plot_with_ROI(image, deformed_landmarks, output_image_filename, bbox_size=15):
    """
    Plot and saves an image with a Region of Interest (ROI) around the predicted landmark position.
    Also saves the ROI coordinates to a .txt file.

    Args:
        - image (numpy.ndarray): The input image to be plotted.
        - deformed_landmarks (numpy.ndarray): An array of landmarks coordinates to be overlaid on the image.
        - output_image_filename (str): The filename to save the resulting plot.
        - bbox_size (int): The size of the bounding box around landmarks. Default is 15
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")

    bbox_coords = []
    counter = 1
    for i in range(0, len(deformed_landmarks), 2):
        x_pred, y_pred = deformed_landmarks[i], deformed_landmarks[i + 1]
        bbox_coords.extend([counter, x_pred, y_pred, bbox_size])
        bounding_box = Rectangle(
            (x_pred - bbox_size / 2, y_pred - bbox_size / 2),
            bbox_size,
            bbox_size,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(bounding_box)
        counter += 1

    # Set axis limits to ensure landmarks and bounding boxes are visible
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    # Save the plot with the ROI printed on it
    plt.savefig(output_image_filename)
    plt.close()

    # Write the bounding box coordinates to a text file and save it
    label_filename = (
        os.path.splitext(os.path.basename(output_image_filename))[0] + ".txt"
    )
    label_path = os.path.join(labels_folder, label_filename)
    with open(label_path, "w") as file:
        file.write("\t x \t y \t bbsize \n")
        for i in range(0, len(bbox_coords), 4):
            file.write("\t".join(map(str, bbox_coords[i : i + 4])) + "\n")


def deform_landmarks(landmarks, displacement_field):
    """
    Deform landmarks based on a displacement field.

    Args:
        landmarks (np.ndarray): An array of shape (N, 2) containing landmark coordinates.
        displacement_field (np.ndarray): The displacement field with shape (1, H, W, 2).

    Returns:
        np.ndarray: Deformed landmarks based on the displacement field.
    """
    deformed_landmarks = []

    for i in range(0, len(landmarks), 2):
        x, y = landmarks[i], landmarks[i + 1]

        # Get the displacement values from the displacement field
        disp_x = displacement_field[0, int(x), int(y), 0]
        disp_y = displacement_field[0, int(x), int(y), 1]

        # Apply displacement to the landmarks
        new_x = x + disp_x
        new_y = y + disp_y

        deformed_landmarks.extend([new_x, new_y])
    return np.array(deformed_landmarks)


######################################## PARAMETERS ########################################

input_folder = (
    "/mnt/c/Users/vitto/Desktop/DL project/Dl project github/processed_dataset1"
)
if not os.path.exists(input_folder):
    input_path = input(
        "Cannot find the dataset path provided in the script.\n"
        "Please input a valid path:"
    )
output_folder = os.getcwd() + "/model_output"


if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    print(f"Folder '{output_folder}' created succesfully in {os.getcwd()}\n")
else:
    print(f"Folder '{output_folder}' already exists in {os.getcwd()}\n")

# Creating new folders called "images" and "labels" inside the output_folder
images_folder = os.path.join(output_folder, "images")
os.makedirs(images_folder, exist_ok=True)

labels_folder = os.path.join(output_folder, "labels")
os.makedirs(labels_folder, exist_ok=True)

# Chose the reference fixed image-label pair
fixed_image_path = "fixed_img.jpg"
fixed_label_path = "fixed_lab.txt"

# Choose which model are you trying
model_name = "disp_pen05.keras"
if not os.path.exists(model_name):
    input_path = input(
        "Cannot find the model path provided in the script.\n"
        "Please input a valid path:"
    )


######################################## MAIN ########################################

os.makedirs(output_folder, exist_ok=True)

displacement_model = tf.keras.models.load_model(model_name, safe_mode=False)

images = glob.glob(os.path.join(input_folder, "**/*.jpg"), recursive=True)

# Processing the fixed data
fixed_image = process_image(fixed_image_path)
fixed_landmarks = load_labels(fixed_label_path)

print("Plotting and saving images and labels...")

for image in tqdm(images):
    moving_image = process_image(image)
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(image))[0]

    # Stack tensors to obtain the correct input for the displacement model
    input = tf.stack([fixed_image, moving_image], -1)
    input = tf.expand_dims(input, axis=0)

    displacement_field = displacement_model.predict(input, verbose=0)
    inverse_displacement = -displacement_field

    # deforming the fixed landmarks onto the moving image (inverse_displacement)
    deformed_landmarks = deform_landmarks(fixed_landmarks, inverse_displacement)

    # Construct the output path with the same filename
    output_path = os.path.join(images_folder, f"{filename}.jpg")

    # Save the normal images in the "images" folder
    # plt.imsave(output_path, moving_image, cmap="gray")

    # Plot and the images with landmarks and ROI
    plot_with_ROI(moving_image, deformed_landmarks, output_path)

print(f"Images and labels saved correctly in {output_folder}\n")
