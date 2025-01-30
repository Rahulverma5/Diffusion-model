import os
import numpy as np
import pandas as pd

import warnings
import tensorflow as tf
from configs.config import IMG_SIZE

warnings.filterwarnings('ignore')

def build_unet(IMG_SIZE=28):
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    output = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

if __name__ == "__main__":
    model =  build_unet()
    model.summary()
    # tf.keras.utils.plot_model(model, show_shapes=True)