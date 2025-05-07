import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *


def squeeze_excitation_layer(input_layer, out_dim, ratio, conv):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_layer)

    excitation = tf.keras.layers.Dense(
        units=int(out_dim / ratio), activation='relu')(squeeze)
    excitation = tf.keras.layers.Dense(
        out_dim, activation='sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape((1, 1, out_dim))(excitation)

    scale = tf.keras.layers.multiply([input_layer, excitation])

    if conv:
        shortcut = tf.keras.layers.Conv2D(out_dim, kernel_size=1, strides=1,
                                          padding='same', kernel_initializer='he_normal')(input_layer)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_layer
    out = tf.keras.layers.Add()([shortcut, scale])
    return out


def conv_block(input_layer, filters):
    layer = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(input_layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(filters*4, kernel_size=1, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    return layer


def squeze_excitation_resnet50_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10,include_top=True):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
    identity_blocks = [3, 4, 6, 3]
    # Block 1
    layer = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1,
                                   padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(x)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    block_1 = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(layer)

    # Block 2
    block_2 = conv_block(block_1, 64)
    block_2 = squeeze_excitation_layer(
        block_2, out_dim=256, ratio=32.0, conv=True)
    for _ in range(identity_blocks[0]-1):
        block_2 = conv_block(block_1, 64)
        block_2 = squeeze_excitation_layer(
            block_2, out_dim=256, ratio=32.0, conv=False)

    # Block 3
    block_3 = conv_block(block_2, 128)
    block_3 = squeeze_excitation_layer(
        block_3, out_dim=512, ratio=32.0, conv=True)
    for _ in range(identity_blocks[1]-1):
        block_3 = conv_block(block_2, 128)
        block_3 = squeeze_excitation_layer(
            block_3, out_dim=512, ratio=32.0, conv=False)

    if include_top:
        pooling = tf.keras.layers.GlobalAveragePooling2D()(block_3)
        model_output = tf.keras.layers.Dense(num_classes,
                                             activation='softmax')(pooling)

        model = tf.keras.models.Model(inputs, model_output)
    else:
        model = tf.keras.models.Model(inputs, block_3)
    return model


if __name__ == "__main__":
    model = squeze_excitation_resnet50_model()
    model.summary(expand_nested=True)