import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *



class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(
            self.channels // self.reduction_ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.channels, activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = self.global_avg_pool(inputs)
        x = tf.reshape(x, (-1, 1, 1, self.channels))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(1, (7, 7), padding='same')

    def call(self, inputs, **kwargs):
        x = tf.nn.sigmoid(self.conv(inputs))
        return x


class ConvolutionBlockAttentionModule(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(ConvolutionBlockAttentionModule, self).__init__(**kwargs)
        self.channels = channels
        self.channel_attention = ChannelAttention(self.channels)
        self.spatial_attention = SpatialAttention()

    def call(self, inputs, **kwargs):
        channel_attention = self.channel_attention(inputs)
        spatial_attention = self.spatial_attention(inputs)
        x = inputs * channel_attention * spatial_attention
        return x


def resnet_block(inputs, filters, strides=1):
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), strides=strides, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if strides != 1:
        shortcut = tf.keras.layers.Conv2D(
            filters, (1, 1), strides=strides)(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    x = ConvolutionBlockAttentionModule(filters)(x)

    return x


def resnet18_cbam_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = resnet_block(x, 64)
    x = resnet_block(x, 128, strides=2)
    x = resnet_block(x, 256, strides=2)
    x = resnet_block(x, 512, strides=2)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, x)
    return model




if __name__ == "__main__":
    model = resnet18_cbam_model()
    model.summary(expand_nested=True)