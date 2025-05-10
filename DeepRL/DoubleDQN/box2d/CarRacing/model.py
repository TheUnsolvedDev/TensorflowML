import tensorflow as tf

from config import *

def QModel(input_shape=ENV_INPUT_SHAPE, output_shape=ENV_OUTPUT_SHAPE):
    he_init = tf.keras.initializers.HeNormal()

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x:x/255.0)(inputs)
    x = tf.keras.layers.Conv2D(
        32, (8, 8), strides=4, activation='relu', kernel_initializer=he_init)(x)
    x = tf.keras.layers.Conv2D(
        64, (4, 4), strides=2, activation='relu', kernel_initializer=he_init)(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), strides=1, activation='relu', kernel_initializer=he_init)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu',
                              kernel_initializer=he_init)(x)
    
    outputs = tf.keras.layers.Dense(
        output_shape[0], kernel_initializer=he_init)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = QModel()
    model.summary()