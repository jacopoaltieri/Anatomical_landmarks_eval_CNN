"""
AUTHORS: Altieri J. , Mazzini V.

This program will train two models to perform cephalometric landmark detection.
The first model is a U-Net to do a coarse detection of each landmark ROI;
The second model is a ResNet50 which will perform the actual keypoint detection.

A data augmentation process is also possible and present as a function, beware that this might be time-consuming.
"""
import os
import matplotlib.pyplot as plt  # ciaone
import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ignore TF unsupported NUMA warnings
import tensorflow as tf


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BATCH_SIZE = 4

######################################## DATA COLLECTION ########################################

# ================================================================ #
#              Importing dataset from directory                    #
# ================================================================ #

print("Started dataset loading...")

# input_path = "/mnt/c/Users/jacop/Desktop/DL_Project/processed_dataset/" #if in wsl
# input_path = r"C:\Users\vitto\Desktop\DL project\DL project github\augmented_dataset\augmented_dataset"  # if in windows
input_path = r"C:\Users\jacop\Desktop\DL_Project\processed_dataset"  # if in windows


# =========== Images =========== #


# Definizione della funzione per l'elaborazione delle immagini
def process_image(x):
    # Leggi il contenuto del file immagine come sequenza di byte
    byte_img = tf.io.read_file(x)
    # Decodifica l'immagine utilizzando il formato JPEG
    img = tf.io.decode_jpeg(byte_img)
    # Convert images to grayscale
    img = tf.image.rgb_to_grayscale(img)
    img = tf.squeeze(img, axis=-1)
    # Normalizza i valori dei pixel nell'intervallo [0, 1]
    img = img / 255
    return img


# Creazione di un dataset per le immagini di addestramento
train_images = tf.data.Dataset.list_files(
    input_path + "/images/train/*.jpg", shuffle=False
)
# Applica la funzione process_image a ciascun elemento del dataset
train_images = train_images.map(process_image)

# Creazione di un dataset per le immagini di test
test_images = tf.data.Dataset.list_files(
    input_path + "/images/test/*.jpg", shuffle=False
)
# Applica la funzione process_image a ciascun elemento del dataset
test_images = test_images.map(process_image)

# Creazione di un dataset per le immagini di validazione
val_images = tf.data.Dataset.list_files(input_path + "/images/val/*.jpg", shuffle=False)
# Applica la funzione process_image a ciascun elemento del dataset
val_images = val_images.map(process_image)


# =========== Labels =========== #


# Definizione di una funzione per caricare le etichette da un file di testo
def load_labels(path):
    # Inizializza una lista vuota per le etichette
    landmarks = []

    # Apre il file di etichette specificato da `path` in modalità di lettura con encoding "utf-8"
    with open(path.numpy(), "r", encoding="utf-8") as file:
        # Salta la prima riga del file
        next(file, None)

        # Itera attraverso le righe rimanenti nel file
        for line in file:
            # Divide ciascuna riga in colonne utilizzando il carattere di tabulazione '\t' come delimitatore
            columns = line.strip().split("\t")

            # Estrae i valori X e Y dalle colonne 1 e 2 (considerando gli indici 0-based)
            # e li converte in valori float. Inoltre, li normalizza dividendo per 256
            x_value = float(columns[1]) / 256
            y_value = float(columns[2]) / 256

            # Aggiunge i valori X e Y normalizzati alla lista `landmarks`
            landmarks.extend([x_value, y_value])

    # Convert `landmarks` to a NumPy array
    return np.array(landmarks)


# Create train, test and val label dataset and load their labels using the corresponding function
train_labels = tf.data.Dataset.list_files(input_path + "/labels/train/*.txt", shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

test_labels = tf.data.Dataset.list_files(input_path + "/labels/test/*.txt", shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

# - `val_labels`: Percorsi dei file di etichette nella directory "val"
val_labels = tf.data.Dataset.list_files(input_path + "/labels/val/*.txt", shuffle=False)
# Mappa la funzione `load_labels` su ciascun percorso di file e ottiene etichette di tipo float16
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))


# =========== Combine Images and Labels =========== #

# Creazione del dataset di addestramento combinando immagini ed etichette
train = tf.data.Dataset.zip((train_images, train_labels))

# Mescola casualmente l'ordine degli elementi nel dataset di addestramento
train = train.shuffle(1500)

# Raggruppa gli elementi del dataset in batch di dimensione specificata da `BATCH_SIZE`
train = train.batch(BATCH_SIZE)

# Implementa il prefetching per caricare in anticipo i dati del prossimo batch
train = train.prefetch(4)

# Lo stesso procedimento viene ora applicato ai dataset di test e validazione

# Creazione del dataset di test combinando immagini ed etichette
test = tf.data.Dataset.zip((test_images, test_labels))

# Mescola casualmente l'ordine degli elementi nel dataset di test
test = test.shuffle(1000)

# Raggruppa gli elementi del dataset in batch di dimensione specificata da `BATCH_SIZE`
test = test.batch(BATCH_SIZE)

# Implementa il prefetching per caricare in anticipo i dati del prossimo batch
test = test.prefetch(4)

# Creazione del dataset di validazione combinando immagini ed etichette
val = tf.data.Dataset.zip((val_images, val_labels))

# Mescola casualmente l'ordine degli elementi nel dataset di validazione
val = val.shuffle(1000)

# Raggruppa gli elementi del dataset in batch di dimensione specificata da `BATCH_SIZE`
val = val.batch(BATCH_SIZE)

# Implementa il prefetching per caricare in anticipo i dati del prossimo batch
val = val.prefetch(4)


"""
# =========== Show some examples =========== #

data_samples = train.as_numpy_iterator()
res = data_samples.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    sample_image = res[0][idx]
    sample_coords = res[1][0][idx]
    
    for i in range(0, len(sample_coords), 2):
        x, y = sample_coords[i], sample_coords[i + 1]
        x_pixel = int(x * 256)
        y_pixel = int(y * 256)
        cv2.circle(sample_image, (x_pixel, y_pixel), 2, (255, 0, 0), -1)
    
    ax[idx].imshow(sample_image,cmap='gray', vmin=0, vmax=1)
plt.show()
"""


######################################## U-NET ########################################

print("Started U-Net training...")

# We only need images in this part
train_images_only = train.map(lambda x, y: x)
test_images_only = test.map(lambda x, y: x)
val_images_only = val.map(lambda x, y: x)

# ================================================================ #
#                        U-Net backbone                            #
# ================================================================ #
# a: activation, c: convolution, p: pooling, u: upconvolution

unet_input_features = 2   # Rappresenta il numero di feature di input (due immagini: fissa e mobile)

# Definizione della forma dell'input
input_shape = (256, 256, 1)
# La forma dell'input sarà un tensore tridimensionale:
# - 256: Larghezza dell'immagine in pixel
# - 256: Altezza dell'immagine in pixel
# - unet_input_features: Numero di feature di input (2 in questo caso)

# Creazione di un layer di input
inputs = tf.keras.layers.Input(input_shape)
# Questo passaggio crea un layer di input utilizzando Keras, che definisce la forma
# dell'input che il modello U-Net accetterà durante l'addestramento e l'inferenza.
# Il layer di input sarà progettato per accettare dati con la forma specificata da input_shape,
# che è adatta per immagini di 256x256 pixel con due feature di input.

### Downsampling path ###

# Applicazione della funzione di attivazione Leaky ReLU con coefficiente alpha
a1 = tf.keras.layers.LeakyReLU(alpha=0.01)(inputs)

# Applicazione di un layer di convoluzione 2D con 64 filtri di dimensione 3x3
c1 = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal")(a1)

# Applicazione del layer di Dropout con un tasso di dropout del 10%
c1 = tf.keras.layers.Dropout(0.1)(c1)

# Applicazione del max-pooling con una finestra di pooling di dimensione 2x2
p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1)


a2 = tf.keras.layers.LeakyReLU(alpha=0.01)(p1)
c2 = tf.keras.layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal")(a2)
c2 = tf.keras.layers.Dropout(0.1)(c2)
p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c2)

a3 = tf.keras.layers.LeakyReLU(alpha=0.01)(p2)
c3 = tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer="he_normal")(a3)
c3 = tf.keras.layers.Dropout(0.2)(c3)
p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c3)

a4 = tf.keras.layers.LeakyReLU(alpha=0.01)(p3)
c4 = tf.keras.layers.Conv2D(512, 3, padding="same", kernel_initializer="he_normal")(a4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

a5 = tf.keras.layers.LeakyReLU(alpha=0.01)(p4)
c5 = tf.keras.layers.Conv2D(1024, 3, padding="same", kernel_initializer="he_normal")(a5)
c5 = tf.keras.layers.Dropout(0.3)(c5)

### Upsampling Path ###

# Decodifica dell'output dell'encoder con la trasposizione della convoluzione 2D
u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = tf.keras.layers.concatenate(
    [u6, c4]
)  # Concatenazione dell'output con gli strati dell'encoder precedenti
a6 = tf.keras.layers.LeakyReLU(alpha=0.01)(u6)  # Applicazione di Leaky ReLU
c6 = tf.keras.layers.Conv2D(512, 3, padding="same", kernel_initializer="he_normal")(
    a6
)  # Convoluzione 2D
c6 = tf.keras.layers.Dropout(0.3)(c6)  # Applicazione di Dropout

u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
a7 = tf.keras.layers.LeakyReLU(alpha=0.01)(u7)
c7 = tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer="he_normal")(a7)
c7 = tf.keras.layers.Dropout(0.2)(c7)

u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
a8 = tf.keras.layers.LeakyReLU(alpha=0.01)(u8)
c8 = tf.keras.layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal")(a8)
c8 = tf.keras.layers.Dropout(0.2)(c8)

u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
a9 = tf.keras.layers.LeakyReLU(alpha=0.01)(u9)
c9 = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal")(a9)
c9 = tf.keras.layers.Dropout(0.1)(c9)

# Creazione dell'output finale del modello
c10 = tf.keras.layers.Conv2D(2, 1, padding="same", kernel_initializer="he_normal")(c9)

# Creazione di un tensore di deformazione
displacement_tensor = tf.keras.layers.Conv2D(
    2, kernel_size=3,activation='linear', padding="same", name="disp"
)(c10)
# Creazione del modello U-Net completo
unet = tf.keras.Model(inputs=[inputs], outputs=[displacement_tensor])

#print("displacement tensor:", displacement_tensor.shape)

unet.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#unet.summary()
unet.fit(train_images_only, epochs=10, validation_data=val_images_only)

# Effettua la fase di test per ottenere l'immagine trasformata
# transformed_image = tf.nn.warpPerspective(test, displacement_tensor)
