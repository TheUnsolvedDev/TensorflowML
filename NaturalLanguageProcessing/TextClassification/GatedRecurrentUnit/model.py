import numpy as np
import tensorflow as tf

from config import *


class DenseEmbedding(tf.keras.layers.Embedding):
    """Embedding layer that forces dense outputs to avoid IndexedSlices in gradients."""
    def call(self, inputs):
        return tf.convert_to_tensor(super().call(inputs))


def LSTM_model(input_shape=(MAX_LEN,), output_shape=(1,), bidirectional=True):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = DenseEmbedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
    if bidirectional:
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32))(x)
    else:
        x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(output_shape[0], activation='sigmoid', dtype='float32')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    model = LSTM_model()
    model.summary()
