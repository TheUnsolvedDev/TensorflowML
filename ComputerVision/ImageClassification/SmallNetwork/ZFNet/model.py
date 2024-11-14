import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *

def model_function(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)
    x = tf.keras.layers.Conv2D(
        64, (7, 7), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=1)(x)
    x = tf.keras.layers.Lambda(
        lambda x: tf.image.per_image_standardization(x))(x)

    x = tf.keras.layers.Conv2D(
        256, (3, 3), strides=1, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
    x = tf.keras.layers.Lambda(
        lambda x: tf.image.per_image_standardization(x))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)

    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(32)(x)

    x = tf.keras.layers.Dense(32)(x)

    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
    model = model_function()
    model.summary(expand_nested=True)