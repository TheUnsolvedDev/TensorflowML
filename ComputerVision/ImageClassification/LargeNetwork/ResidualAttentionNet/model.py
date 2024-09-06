import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *




def attention_module(x, n_filters):
    skip = x

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(n_filters, kernel_size=(
        1, 1), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(n_filters, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(n_filters, kernel_size=(
        1, 1), strides=(1, 1), padding='same')(x)

    x = tf.keras.layers.Multiply()([x, skip])
    x = tf.keras.layers.Add()([x, skip])

    return x


def residual_block(x, n_filters):
    skip = x

    x = tf.keras.layers.Conv2D(n_filters, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = attention_module(x, n_filters)
    x = tf.keras.layers.Conv2D(n_filters, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Concatenate(axis=-1)([x, skip])

    return x


def residual_attention_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    model = residual_attention_model()
    model.summary(expand_nested=True)