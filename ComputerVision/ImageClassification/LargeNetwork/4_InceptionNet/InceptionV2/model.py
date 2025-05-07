import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *



def conv_block(x, filters, kernel_size, strides=1, padding='same', activation=True):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x) if activation else x

def inception_block_a(x):
    branch1 = conv_block(x, 64, (1, 1))
    
    branch2 = conv_block(x, 48, (1, 1))
    branch2 = conv_block(branch2, 64, (5, 5))
    
    branch3 = conv_block(x, 64, (1, 1))
    branch3 = conv_block(branch3, 96, (3, 3))
    branch3 = conv_block(branch3, 96, (3, 3))
    
    branch4 = tf.keras.layers.AveragePooling2D((3, 3), strides=1, padding='same')(x)
    branch4 = conv_block(branch4, 32, (1, 1))
    
    return tf.keras.layers.concatenate([branch1, branch2, branch3, branch4])

def inception_block_b(x):
    branch1 = conv_block(x, 384, (3, 3), strides=2, padding='valid')
    
    branch2 = conv_block(x, 64, (1, 1))
    branch2 = conv_block(branch2, 96, (3, 3))
    branch2 = conv_block(branch2, 96, (3, 3), strides=2, padding='valid')
    
    branch3 = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='valid')(x)
    
    return tf.keras.layers.concatenate([branch1, branch2, branch3])

def inception_block_c(x):
    branch1 = conv_block(x, 192, (1, 1))
    
    branch2 = conv_block(x, 128, (1, 1))
    branch2 = conv_block(branch2, 160, (1, 7))
    branch2 = conv_block(branch2, 192, (7, 1))
    
    branch3 = conv_block(x, 128, (1, 1))
    branch3 = conv_block(branch3, 160, (7, 1))
    branch3 = conv_block(branch3, 160, (1, 7))
    branch3 = conv_block(branch3, 160, (7, 1))
    branch3 = conv_block(branch3, 192, (1, 7))
    
    branch4 = tf.keras.layers.AveragePooling2D((3, 3), strides=1, padding='same')(x)
    branch4 = conv_block(branch4, 192, (1, 1))
    
    return tf.keras.layers.concatenate([branch1, branch2, branch3, branch4])

def inception2_model(input_shape=[299, 299, 3], num_classes=10):
    # input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255.)(inputs)
    
    x = conv_block(x, 32, (3, 3), strides=2, padding='valid')
    x = conv_block(x, 32, (3, 3), strides=1, padding='valid')
    x = conv_block(x, 64, (3, 3), strides=1, padding='same')
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='valid')(x)
    
    x = conv_block(x, 80, (3, 3), strides=1, padding='valid')
    x = conv_block(x, 192, (3, 3), strides=2, padding='valid')
    x = conv_block(x, 288, (3, 3), strides=1, padding='same')
    
    for _ in range(2):
        x = inception_block_a(x)
    
    for _ in range(3):
        x = inception_block_b(x)
    
    for _ in range(2):
        x = inception_block_c(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
    model = inception2_model()
    model.summary(expand_nested=True)