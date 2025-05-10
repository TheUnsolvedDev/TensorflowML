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
    
    v = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=he_init)(x)
    v = tf.keras.layers.Dense(1, kernel_initializer=he_init)(v)
    
    # Advantage stream
    a = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=he_init)(x)
    a = tf.keras.layers.Dense(output_shape[0], kernel_initializer=he_init)(a)
    
    # Combine using a Lambda layer
    def dueling_combine(inputs):
        v, a = inputs
        return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
    
    outputs = tf.keras.layers.Lambda(dueling_combine)([v, a])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = QModel()
    model.summary()