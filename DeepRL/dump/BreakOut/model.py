import tensorflow as tf

from config import *

def QModel(input_shape=(IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS), num_actions=NUM_ACTIONS):
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
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_actions)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = QModel()
    model.summary()
    # Test the model with a random input
    test_input = tf.random.normal((1, IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS))
    test_output = model(test_input)
    print("Test output shape:", test_output.shape)
    print("Test output:", test_output.numpy())