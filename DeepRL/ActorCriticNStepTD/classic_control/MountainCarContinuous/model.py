import tensorflow as tf
import numpy as np

from config import *

def ActorModel(input_shape=ENV_INPUT_SHAPE, output_shape=ENV_OUTPUT_SHAPE):
    he_init = tf.keras.initializers.HeNormal()
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(2**input_shape[0], activation='selu')(inputs)
    x = tf.keras.layers.Dense(4**output_shape[0], activation='selu')(x)
    mu = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=he_init)(x)
    std = tf.keras.layers.Dense(1, activation="softplus", kernel_initializer=he_init)(x)
    model = tf.keras.Model(inputs=inputs, outputs=[mu, std])
    return model

def CriticModel(input_shape=ENV_INPUT_SHAPE, output_shape=(1,)):
    he_init = tf.keras.initializers.HeNormal()
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(2**input_shape[0], activation='selu')(inputs)
    x = tf.keras.layers.Dense(4**output_shape[0], activation='selu')(x)
    outputs = tf.keras.layers.Dense(output_shape[0], kernel_initializer=he_init)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model = ActorModel()
    model.summary()
    x = np.random.rand(1, ENV_INPUT_SHAPE[0])
    y = model.predict(x)
    print(y)