import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import string
import re
import os

import argparse

parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU number')
args = parser.parse_args()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[args.gpu], 'GPU')


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_1d_cnn_text_classifier.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

BATCH_SIZE = 128
MAX_FEATURES = 20000
EMBEDDING_DIM = 128
MAX_LENGTH = 500


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )


class Dataset:
    def __init__(self,):
        batch_size = BATCH_SIZE

        self.raw_train_ds = tf.keras.utils.text_dataset_from_directory(
            "aclImdb/train",
            batch_size=batch_size,
            validation_split=0.2,
            subset="training",
            seed=1337,
        )
        self.raw_val_ds = tf.keras.utils.text_dataset_from_directory(
            "aclImdb/train",
            batch_size=batch_size,
            validation_split=0.2,
            subset="validation",
            seed=1337,
        )
        self.raw_test_ds = tf.keras.utils.text_dataset_from_directory(
            "aclImdb/test", batch_size=batch_size
        )

    def get_data(self):
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=custom_standardization,
            max_tokens=MAX_FEATURES,
            output_mode="int",
            output_sequence_length=MAX_LENGTH,
        )
        text_ds = self.raw_train_ds.map(lambda x, y: x)
        self.vectorize_layer.adapt(text_ds)

        def vectorize_text(text, label):
            text = tf.expand_dims(text, -1)
            return self.vectorize_layer(text), label

        train_ds = self.raw_train_ds.map(vectorize_text)
        val_ds = self.raw_val_ds.map(vectorize_text)
        test_ds = self.raw_test_ds.map(vectorize_text)
        train_ds = train_ds.cache().prefetch(buffer_size=10)
        val_ds = val_ds.cache().prefetch(buffer_size=10)
        test_ds = test_ds.cache().prefetch(buffer_size=10)

        return train_ds, val_ds, test_ds


def CNNModel1D():
    inputs = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype="int64")
    x = tf.keras.layers.Embedding(MAX_FEATURES, EMBEDDING_DIM)(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv1D(
        128, 7, padding="valid", activation="relu", strides=3)(x)
    x = tf.keras.layers.Conv1D(
        128, 7, padding="valid", activation="relu", strides=3)(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(
        1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)
    return model


if __name__ == '__main__':
    dataset = Dataset()
    train_ds, val_ds, test_ds = dataset.get_data()
    model = CNNModel1D()
    tf.keras.utils.plot_model(model, to_file="model_1d_cnn_text_classifier.png",
                              show_shapes=True, expand_nested=True, show_layer_activations=True)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)
    model.evaluate(test_ds)
