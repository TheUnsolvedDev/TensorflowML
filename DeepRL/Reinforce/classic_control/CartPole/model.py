import tensorflow as tf
import numpy as np

from config import *

def PolicyModel(input_shape=ENV_INPUT_SHAPE, output_shape=ENV_OUTPUT_SHAPE):
    he_init = tf.keras.initializers.HeNormal()
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=he_init)(inputs)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=he_init)(x)
    outputs = tf.keras.layers.Dense(output_shape[0], activation='softmax', kernel_initializer=he_init)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def BaselineModel(input_shape=ENV_INPUT_SHAPE, output_shape=(1,)):
    he_init = tf.keras.initializers.HeNormal()
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=he_init)(inputs)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=he_init)(x)
    outputs = tf.keras.layers.Dense(output_shape[0], kernel_initializer=he_init)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model = PolicyModel()
    model.summary()
    x = np.random.rand(1, ENV_INPUT_SHAPE[0])
    y = model.predict(x)
    print(y)
    
    baseline_model = BaselineModel()
    baseline_model.summary()