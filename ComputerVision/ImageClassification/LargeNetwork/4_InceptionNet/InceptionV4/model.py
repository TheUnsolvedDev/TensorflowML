import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *

# --- BasicConv2D ---
class BasicConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), strides=1, padding='valid'):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.relu(x)

# --- Inception-A ---
def InceptionA(x):
    b1 = BasicConv2D(96, (1, 1))(x)

    b2 = BasicConv2D(64, (1, 1))(x)
    b2 = BasicConv2D(96, (3, 3), padding='same')(b2)

    b3 = BasicConv2D(64, (1, 1))(x)
    b3 = BasicConv2D(96, (3, 3), padding='same')(b3)
    b3 = BasicConv2D(96, (3, 3), padding='same')(b3)

    b4 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    b4 = BasicConv2D(96, (1, 1))(b4)

    return tf.keras.layers.Concatenate(axis=-1)([b1, b2, b3, b4])

# --- Reduction-A ---
def ReductionA(x, k=192, l=224, m=256, n=384):
    b1 = BasicConv2D(n, (3, 3), strides=2, padding='valid')(x)

    b2 = BasicConv2D(k, (1, 1))(x)
    b2 = BasicConv2D(l, (3, 3), padding='same')(b2)
    b2 = BasicConv2D(m, (3, 3), strides=2, padding='valid')(b2)

    b3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    return tf.keras.layers.Concatenate(axis=-1)([b1, b2, b3])

# --- Inception-B ---
def InceptionB(x):
    b1 = BasicConv2D(384, (1, 1))(x)

    b2 = BasicConv2D(192, (1, 1))(x)
    b2 = BasicConv2D(224, (1, 7), padding='same')(b2)
    b2 = BasicConv2D(256, (7, 1), padding='same')(b2)

    b3 = BasicConv2D(192, (1, 1))(x)
    b3 = BasicConv2D(192, (7, 1), padding='same')(b3)
    b3 = BasicConv2D(224, (1, 7), padding='same')(b3)
    b3 = BasicConv2D(224, (7, 1), padding='same')(b3)
    b3 = BasicConv2D(256, (1, 7), padding='same')(b3)

    b4 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    b4 = BasicConv2D(128, (1, 1))(b4)

    return tf.keras.layers.Concatenate(axis=-1)([b1, b2, b3, b4])

# --- Reduction-B ---
def ReductionB(x):
    b1 = BasicConv2D(192, (1, 1))(x)
    b1 = BasicConv2D(192, (3, 3), strides=2, padding='valid')(b1)

    b2 = BasicConv2D(256, (1, 1))(x)
    b2 = BasicConv2D(256, (1, 7), padding='same')(b2)
    b2 = BasicConv2D(320, (7, 1), padding='same')(b2)
    b2 = BasicConv2D(320, (3, 3), strides=2, padding='valid')(b2)

    b3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    return tf.keras.layers.Concatenate(axis=-1)([b1, b2, b3])

# --- Inception-C ---
def InceptionC(x):
    b1 = BasicConv2D(256, (1, 1))(x)

    b2 = BasicConv2D(384, (1, 1))(x)
    b2a = BasicConv2D(256, (1, 3), padding='same')(b2)
    b2b = BasicConv2D(256, (3, 1), padding='same')(b2)
    b2 = tf.keras.layers.Concatenate(axis=-1)([b2a, b2b])

    b3 = BasicConv2D(384, (1, 1))(x)
    b3 = BasicConv2D(448, (3, 1), padding='same')(b3)
    b3 = BasicConv2D(512, (1, 3), padding='same')(b3)
    b3a = BasicConv2D(256, (3, 1), padding='same')(b3)
    b3b = BasicConv2D(256, (1, 3), padding='same')(b3)
    b3 = tf.keras.layers.Concatenate(axis=-1)([b3a, b3b])

    b4 = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    b4 = BasicConv2D(256, (1, 1))(b4)

    return tf.keras.layers.Concatenate(axis=-1)([b1, b2, b3, b4])



def inception4_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10, A=4, B=7, C=3):
    inputs = tf.keras.Input(shape=input_shape)

    # Stem
    x = tf.keras.layers.Lambda(lambda x: x / 255.)(inputs)
    x = BasicConv2D(32, (3, 3), strides=2, padding='valid')(x)
    x = BasicConv2D(32, (3, 3), padding='valid')(x)
    x = BasicConv2D(64, (3, 3))(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='valid')(x)

    x = BasicConv2D(80, (1, 1))(x)
    x = BasicConv2D(192, (3, 3), padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='valid')(x)

    # Inception-A blocks
    for _ in range(A):
        x = InceptionA(x)

    # Reduction-A
    x = ReductionA(x)

    # Inception-B blocks
    for _ in range(B):
        x = InceptionB(x)

    # Reduction-B
    x = ReductionB(x)

    # Inception-C blocks
    for _ in range(C):
        x = InceptionC(x)

    # Final layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name="InceptionV4")


if __name__ == "__main__":
    model = inception4_model()
    model.summary(expand_nested=True)
