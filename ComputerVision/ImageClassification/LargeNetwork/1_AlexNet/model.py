import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *

class LocalResponseNormalization(tf.keras.layers.Layer):
    def __init__(self, alpha=0.0001, beta=0.75, depth_radius=5, **kwargs):
        super(LocalResponseNormalization, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.depth_radius = depth_radius

    def build(self, input_shape):
        self.channels = input_shape[-1]  # Get the number of channels
        self.kernel = self.add_weight(
            shape=(1, 1, self.channels, 1),
            initializer=tf.keras.initializers.Ones(),
            trainable=False
        )

    def call(self, x):
        squared = tf.square(x)
        window_sum = tf.nn.depthwise_conv2d(
            squared,
            self.kernel,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )
        norm = tf.pow(1 + self.alpha * window_sum, -self.beta)
        return x * norm

    def compute_output_shape(self, input_shape):
        return input_shape


def alexnet_model(input_shape=[227, 227, 3], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)
    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4))(x)
    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(4, 4), padding="same")(x)
    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(4, 4), activation='relu', padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(4, 4), activation='relu', padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(4, 4), activation='relu', padding="same")(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


if __name__ == '__main__':
    model = alexnet_model()
    model.summary()

    tf.keras.utils.plot_model(
        model, to_file=alexnet_model.__name__+'.png', show_shapes=True)