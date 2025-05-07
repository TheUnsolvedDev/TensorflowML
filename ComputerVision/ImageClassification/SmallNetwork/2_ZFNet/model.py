import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *

import tensorflow as tf


def zfnet_model(input_shape, num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)
    x = tf.keras.layers.Conv2D(96, (7, 7), strides=(
        2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(256, (5, 5), strides=(
        2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(
        384, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        384, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


if __name__ == "__main__":
    model = zfnet_model()
    model.summary(expand_nested=True)
