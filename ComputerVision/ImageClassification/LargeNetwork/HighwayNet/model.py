import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *


def highway_network_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x:x/255)(inputs)

    # Convolutional tf.keras.layers
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Highway layers
    for i in range(4):
        H = tf.keras.layers.Dense(64, activation='relu')(x)
        T = tf.keras.layers.Dense(64, activation='sigmoid')(x)
        x = tf.keras.layers.Lambda(
            lambda inputs: inputs[0] * inputs[1] + inputs[0] * (1 - inputs[1]))([H, T])

    # Fully connected layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = highway_network_model()
    model.summary(expand_nested=True)