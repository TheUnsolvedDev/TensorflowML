import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *

def conv_bn(x, filters, kernel_size, strides=1):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def sep_bn(x, filters, kernel_size, strides=1):
    x = tf.keras.layers.SeparableConv2D(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding='same',
                                        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def entry_flow(x):

    x = conv_bn(x, filters=32, kernel_size=3, strides=2)
    x = tf.keras.layers.ReLU()(x)
    x = conv_bn(x, filters=64, kernel_size=3, strides=1)
    tensor = tf.keras.layers.ReLU()(x)

    x = sep_bn(tensor, filters=128, kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=128, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])

    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])

    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])
    return x


def middle_flow(tensor):

    for _ in range(8):
        x = tf.keras.layers.ReLU()(tensor)
        x = sep_bn(x, filters=256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = sep_bn(x, filters=256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        x = sep_bn(x, filters=256, kernel_size=3)
        x = tf.keras.layers.ReLU()(x)
        tensor = tf.keras.layers.Add()([tensor, x])

    return tensor


def exit_flow(tensor,output_shape):

    x = tf.keras.layers.ReLU()(tensor)
    x = sep_bn(x, filters=256,  kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=512,  kernel_size=3)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=512, kernel_size=1, strides=2)
    x = tf.keras.layers.Add()([tensor, x])

    x = sep_bn(x, filters=512,  kernel_size=3)
    x = tf.keras.layers.ReLU()(x)
    x = sep_bn(x, filters=512,  kernel_size=3)
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    x = tf.keras.layers.Dense(units=10, activation='softmax')(x)

    return x

def model_function(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)
    x = entry_flow(x)
    x = middle_flow(x)
    output = exit_flow(x,num_classes)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

if __name__ == "__main__":
    model = model_function()
    model.summary(expand_nested=True)