import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *


def conv_block(net_in, K, dropout_rate, weight_decay):
    net = tf.keras.layers.BatchNormalization()(net_in)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Conv2D(4*K, (1, 1), use_bias=False, padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(net)
    net = tf.keras.layers.Dropout(dropout_rate)(net)

    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Conv2D(K, (3, 3), use_bias=False, padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(net)
    net = tf.keras.layers.Dropout(dropout_rate)(net)

    net = tf.keras.layers.Concatenate()([net_in, net])
    return net


def dense_block(net_in, num_blocks, K, dropout_rate, weight_decay):
    net = net_in
    for _ in range(num_blocks):
        net = conv_block(net, K, dropout_rate, weight_decay)
    return net


def transition_block(net_in, theta, dropout_rate, weight_decay):
    net = tf.keras.layers.BatchNormalization()(net_in)
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Conv2D(int(net_in.shape[-1]*theta), (1, 1), use_bias=False, padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(net)
    net = tf.keras.layers.Dropout(dropout_rate)(net)
    net = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(net)
    return net


def densenet_model_201(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]), num_classes=10, K=64, theta=0.5, num_blocks=[6, 12, 48, 32], dropout_rate=0.4, weight_decay=1e-4):
    if theta <= 0.0 or theta > 1.0:
        raise Exception('Compression factor must be > 0 and <= 1.0')
    if dropout_rate <= 0.0 or dropout_rate > 1.0:
        raise Exception('Drop rate must be > 0 and <= 1.0')

    inputs = tf.keras.layers.Input(shape=input_shape)
    net = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

    net = tf.keras.layers.Conv2D(2*K,(7, 7), (2, 2), use_bias=False, padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(net)
    net = tf.keras.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same')(net)

    for i in range(len(num_blocks) - 1):
        net = dense_block(net,num_blocks[i],K,dropout_rate,weight_decay)
        net = transition_block(net,theta,dropout_rate,weight_decay)
    net = dense_block(net,num_blocks[-1],K,dropout_rate,weight_decay)

    net = tf.keras.layers.GlobalAveragePooling2D()(net)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(net)
    model = tf.keras.models.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = densenet_model_201()
    model.summary(expand_nested=True)
