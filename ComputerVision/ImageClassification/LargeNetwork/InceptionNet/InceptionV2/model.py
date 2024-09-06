import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *


def inception_module(x, f1, f3_reduce, f3, f5_reduce, f5, pool_proj):
    conv1 = tf.keras.layers.Conv2D(
        f1, (1, 1), padding='same', activation='relu')(x)
    conv3_reduce = tf.keras.layers.Conv2D(
        f3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv3 = tf.keras.layers.Conv2D(
        f3, (3, 3), padding='same', activation='relu')(conv3_reduce)
    conv5_reduce = tf.keras.layers.Conv2D(
        f5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv5 = tf.keras.layers.Conv2D(
        f5, (5, 5), padding='same', activation='relu')(conv5_reduce)
    pool = tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = tf.keras.layers.Conv2D(
        pool_proj, (1, 1), padding='same', activation='relu')(pool)
    output = tf.keras.layers.Concatenate(axis=-1)(
        [conv1, conv3, conv5, pool_proj])
    return output


def inception_B(x):
    # 1x1 convolution
    conv1 = tf.keras.layers.Conv2D(
        192, (1, 1), padding='same', activation='relu')(x)
    conv2 = tf.keras.layers.Conv2D(
        192, (1, 1), padding='same', activation='relu')(x)
    conv2 = tf.keras.layers.Conv2D(
        192, (1, 7), padding='same', activation='relu')(conv2)
    conv2 = tf.keras.layers.Conv2D(
        192, (7, 1), padding='same', activation='relu')(conv2)
    output = tf.keras.layers.Concatenate(axis=-1)([conv1, conv2])
    return output


def inception_C(x):
    conv1 = tf.keras.layers.Conv2D(
        320, (1, 1), padding='same', activation='relu')(x)
    conv2_reduce = tf.keras.layers.Conv2D(
        384, (1, 1), padding='same', activation='relu')(x)
    conv2_1 = tf.keras.layers.Conv2D(
        384, (1, 3), padding='same', activation='relu')(conv2_reduce)
    conv2_2 = tf.keras.layers.Conv2D(
        384, (3, 1), padding='same', activation='relu')(conv2_reduce)
    conv2 = tf.keras.layers.Concatenate(axis=-1)([conv2_1, conv2_2])
    output = tf.keras.layers.Concatenate(axis=-1)([conv1, conv2])
    return output


def linear_bottleneck(x, filters, strides):
    x = tf.keras.layers.Conv2D(
        filters, (1, 1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), strides=strides, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(
        filters, (1, 1), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    shortcut = tf.keras.layers.Conv2D(
        filters, (1, 1), strides=strides, padding='same', use_bias=False)(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def inception2_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    # input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255.)(inputs)
    # stem module
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Inception modules
    x = inception_module(x, f1=16, f3_reduce=16, f3=16,
                         f5_reduce=16, f5=8, pool_proj=8)
    x = inception_module(x, f1=32, f3_reduce=32, f3=32,
                         f5_reduce=8, f5=16, pool_proj=16)
    x = inception_B(x)
    x = inception_module(x, f1=128, f3_reduce=32, f3=128,
                         f5_reduce=24, f5=16, pool_proj=16)
    x = inception_module(x, f1=128, f3_reduce=32, f3=128,
                         f5_reduce=24, f5=16, pool_proj=16)
    x = inception_module(x, f1=128, f3_reduce=32, f3=128,
                         f5_reduce=24, f5=16, pool_proj=16)
    x = inception_module(x, f1=128, f3_reduce=128, f3=128,
                         f5_reduce=48, f5=32, pool_proj=32)
    x = inception_C(x)
    x = inception_module(x, f1=128, f3_reduce=128, f3=128,
                         f5_reduce=48, f5=32, pool_proj=32)
    x = inception_module(x, f1=128, f3_reduce=128, f3=128,
                         f5_reduce=48, f5=32, pool_proj=32)

    # top layer
    x = tf.keras.layers.AveragePooling2D((4, 4))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # create and compile the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    model = inception2_model()
    model.summary(expand_nested=True)