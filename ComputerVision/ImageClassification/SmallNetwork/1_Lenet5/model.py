import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *

def lenet5_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)
    x = tf.keras.layers.Conv2D(
        filters=6, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(120, activation='relu')(x)
    x = tf.keras.layers.Dense(84, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model_fn = tf.keras.Model(inputs=inputs, outputs=x)
    return model_fn

if __name__ == "__main__":
    model = lenet5_model()
    model.summary(expand_nested=True)