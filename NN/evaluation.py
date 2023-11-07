"""
AUTHORS: Altieri J. , Mazzini V.

"""
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from tqdm import tqdm
import glob

warnings.filterwarnings("ignore")  # suppress tfa deprecated warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ignore TF unsupported NUMA warnings

import tensorflow as tf
import tensorflow_addons as tfa

warnings.resetwarnings()  # restore warnings


######################################## FUNCTIONS ########################################

def process_image(path):
    """Reads the file as a sequence of bytes and converts it
    into a grayscale JPG image normalized in [0,1]

    Args:
        x (str): path to the image

    Returns:
        img: processed image
    """
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (256,256))
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
    net_feed = tf.stack([fixed_image, moving_image], -1)
    net_feed = tf.expand_dims(net_feed, axis=0)
    return net_feed


def plot_with_landmarks_and_ROI(image, deformed_landmarks, output_image_filename, bbox_size=15):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")

    # Create a bounding box around the deformed landmarks in red
    for i in range(0, len(deformed_landmarks), 2):
        x_pred, y_pred = deformed_landmarks[i], deformed_landmarks[i + 1]

        bounding_box = Rectangle(
            (x_pred - bbox_size / 2, y_pred - bbox_size / 2),
            bbox_size,
            bbox_size,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(bounding_box)

    # Set axis limits to ensure landmarks and bounding boxes are visible
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)




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

input_folder='/mnt/c/Users/jacop/Desktop/DL_Project/processed_dataset'

#input_folder = input("Dataset you want to analyze:")
output_folder = os.getcwd()+"/model_output"

# Chose the reference fixed image-label pair
fixed_image_path = "fixed_img.jpg"
fixed_label_path = "fixed_lab.txt"

model_name = "disp_pen07.keras"


######################################## MAIN ########################################

os.makedirs(output_folder, exist_ok=True)

displacement_model = tf.keras.models.load_model(model_name, safe_mode=False)

images = glob.glob(os.path.join(input_folder, '**/*.jpg'), recursive=True)


# Processing the fixed data
fixed_image = process_image(fixed_image_path)
fixed_landmarks = load_labels(fixed_label_path)


for image in tqdm(images):
    moving_image = process_image(image)
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(image))[0]
    

    # Stack tensors to obtain the correct input for the displacement model
    input = tf.stack([fixed_image, moving_image], -1)
    input = tf.expand_dims(input, axis=0)
    
    displacement_field = displacement_model.predict(input, verbose = 0)
    inverse_displacement = -displacement_field    
    
    # deforming the fixed landmarks onto the moving image (inverse_displacement)
    deformed_landmarks = deform_landmarks(fixed_landmarks,inverse_displacement)
    
    # Construct the output path with the same filename
    output_path = os.path.join(output_folder, f'{filename}.jpg')
    

#     #TODO: PLOT AND SAVE THE IMAGES IN OUTPUT FOLDER/IMAGES
    
# # TODO: WRITE BBOX COORDS IN TXT FILES: landmark x y bboxsize
