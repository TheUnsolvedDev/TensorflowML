import tensorflow as tf

from config import *

def DuelingQModel(input_shape=(IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS), num_actions=NUM_ACTIONS):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)  # Normalize the input
    x = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)

    # Dueling architecture
    # Value stream
    v = tf.keras.layers.Dense(64, activation='relu')(x)
    v = tf.keras.layers.Dense(1)(v)

    # Advantage stream
    a = tf.keras.layers.Dense(64, activation='relu')(x)
    a = tf.keras.layers.Dense(num_actions)(a)

    # Combine streams
    def combine_streams(inputs):
        v, a = inputs
        return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))

    q_values = tf.keras.layers.Lambda(combine_streams)([v, a])

    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    return model

if __name__ == "__main__":
    model = DuelingQModel()
    model.summary()
    test_input = tf.random.normal((1, IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS))
    test_output = model(test_input)
    print("Test output shape:", test_output.shape)
    print("Test output:", test_output.numpy())
