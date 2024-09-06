import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *



num_blocks = 3
num_layers_per_block = 4
growth_rate = 16
dropout_rate = 0.4
compress_factor = 0.5
eps = 1.1e-5
num_filters = 16


def H(inputs, num_filters, dropout_rate):
    x = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(
        3, 3), use_bias=False, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    return x


def transition(inputs, num_filters, compression_factor, dropout_rate):
    # compression_factor is the 'θ'
    x = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    num_feature_maps = inputs.shape[1]  # The value of 'm'

    x = tf.keras.layers.Conv2D(np.floor(compression_factor * num_feature_maps).astype(np.int32),
                               kernel_size=(1, 1), use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    return x


def dense_block(inputs, num_layers, num_filters, growth_rate, dropout_rate):
    for i in range(num_layers):  # num_layers is the value of 'l'
        conv_outputs = H(inputs, num_filters, dropout_rate)
        inputs = tf.keras.layers.Concatenate()([conv_outputs, inputs])
        # To increase the number of filters for each layer.
        num_filters += growth_rate
    return inputs, num_filters


def densenet_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    global num_filters
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(
        3, 3), use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    for i in range(num_blocks):
        x, num_filters = dense_block(
            x, num_layers_per_block, num_filters, growth_rate, dropout_rate)
        x = transition(x, num_filters, compress_factor, dropout_rate)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = densenet_model()
    model.summary(expand_nested=True)