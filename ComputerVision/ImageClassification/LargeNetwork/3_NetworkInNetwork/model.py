import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *

# Define the NiN block
def NiN_block(inputs, filters, kernel_size, strides, padding):
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

# Define the NiN model


def network_in_network_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
    x = NiN_block(x, filters=192, kernel_size=5,
                  strides=1, padding='same')
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = NiN_block(x, filters=160, kernel_size=1, strides=1, padding='same')
    x = NiN_block(x, filters=96, kernel_size=1, strides=1, padding='same')
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    model = network_in_network_model()
    model.summary(expand_nested=True)