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


def vgg11_A_LRN_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                              padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    models = [
        vgg11_A_LRN_model,]
    
    for model_fn in models:
        model = model_fn()
        model.summary()