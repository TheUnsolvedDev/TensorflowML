import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *

def binary(x):
    return tf.where(x >= 0, tf.ones_like(x), -tf.ones_like(x))


def model_function(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)
    x = tf.keras.layers.Conv2D(
        32, 3, activation=None, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Lambda(binary)(x)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(
        64, 3, activation=None, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Lambda(binary)(x)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(
        128, 3, activation=None, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Lambda(binary)(x)
    x = tf.keras.layers.Activation("tanh")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    xnornet = tf.keras.Model(inputs=inputs, outputs=x)
    return xnornet

if __name__ == "__main__":
    model = model_function()
    model.summary(expand_nested=True)