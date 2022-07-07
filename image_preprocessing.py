from tkinter import Image
from unicodedata import decimal
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os

def load_data(directory, image_size, subset = None, val_split = None, seed = 1024, shuffle = False, batch_size = 32, interpolation='nearest', autotune=False):
    def set_seed(seed=31415):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    set_seed()


    data = keras.preprocessing.image_dataset_from_directory(
        directory = directory,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=val_split,
        subset=subset,
        interpolation = interpolation, 
    )

    def convert_to_float(image, label):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, label

    if autotune == True:
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data = data.map(convert_to_float).cache().prefetch(buffer_size=AUTOTUNE)
    return data

def show_img(dataset, figsize, nx, ny):
    class_names = dataset.class_names

    plt.figure(figsize=figsize, )
    for images, labels in dataset.take(1):
        for i in range(nx*ny):
            ax = plt.subplot(nx, ny, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

def data_augmenter_layer(augmenter_list):
    data_augmentation = keras.Sequential()
    for i in range(len(augmenter_list)):
        data_augmentation.add(augmenter_list[i])

    return data_augmentation
    
def data_rescaling(value):
    rescale = keras.Sequential(
        layers.Rescaling(value)
    )

    return rescale

def read_image(path, size, scaled=True, newaxis=False, tensor=True):
    img = tf.keras.preprocessing.image.load_img(path, target_size=size)
    img = np.array(img)

    if scaled :
        img = np.around(np.array(img)/255.0, decimals=12)

    if newaxis:
        img = img[np.newaxis, ...]

    if tensor:
        img = tf.constant(img)

    return img


def decompress_from_zip(_URL, name, extract=True):
    zip_dir = tf.keras.utils.get_file(name, origin=_URL, extract=extract)

    return zip_dir

### Notes :
"""
    img_gen = ImageDataGenerator()
    train_data = img_gen.flow_from_directory()
    model.fit_generator()

"""