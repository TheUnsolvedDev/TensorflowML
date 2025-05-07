import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *




def conv_block(inputs, filters, kernel_size, strides, cardinality):
    group_channels = filters // cardinality
    groups = tf.keras.layers.Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': cardinality})(inputs)
    # groups = tf.split(inputs, cardinality, axis=-1)
    conv_outputs = []
    for i in range(cardinality):
        conv_outputs.append(tf.keras.layers.Conv2D(
            filters=group_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding='same')(groups[i]))
    concat = tf.keras.layers.Concatenate(axis=-1)(conv_outputs)
    outputs = tf.keras.layers.BatchNormalization()(concat)
    outputs = tf.keras.layers.Activation('relu')(outputs)
    return outputs

def resnext_block(inputs, filters, kernel_size, strides, cardinality):
    conv_outputs = conv_block(inputs, filters, kernel_size, strides, cardinality)
    if inputs.shape[-1] != filters:
        skip_connection = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=strides,
            padding='same')(inputs)
        skip_connection = tf.keras.layers.BatchNormalization()(skip_connection)
    else:
        skip_connection = inputs
    outputs = tf.keras.layers.Add()([conv_outputs, skip_connection])
    outputs = tf.keras.layers.Activation('relu')(outputs)
    return outputs

def resnext_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # ResNeXt blocks
    x = resnext_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2), cardinality=32)
    x = resnext_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2), cardinality=32)
    x = resnext_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2), cardinality=32)
    
    # Global average pooling and dense layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)
    
    # Create the model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    model = resnext_model()
    model.summary(expand_nested=True)