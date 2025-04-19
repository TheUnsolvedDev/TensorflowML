import tensorflow as tf

from config import *

def QModel(input_shape=ENV_INPUT_SHAPE, output_shape=ENV_OUTPUT_SHAPE):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='selu')(inputs)
    x = tf.keras.layers.Dense(64, activation='selu')(x)
    outputs = tf.keras.layers.Dense(output_shape[0], activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = QModel()
    model.summary()