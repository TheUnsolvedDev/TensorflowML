import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *




def vgg16_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


if __name__ == "__main__":
    model = vgg16_model()
    model.summary(expand_nested=True)