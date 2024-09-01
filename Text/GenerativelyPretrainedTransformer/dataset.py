import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import pickle
import os

from config import *
from text_preprocessing import DataCleaning


class Dataset:
    def __init__(self):
        self.clean = DataCleaning('Dataset/reddit_short_stories_cleaned.txt')
        self.dataset = self.clean.create_dataset()

        if not os.path.exists('tv_layer.pkl'):
            self.layer = tf.keras.layers.TextVectorization(
                standardize=None,
                max_tokens=VOCAB_SIZE-1,
                output_mode='int',
                output_sequence_length=MAX_LENGTH+1
            )
            self.layer.adapt(self.dataset)
            self.vocab = self.layer.get_vocabulary()
            pickle.dump({'config': self.layer.get_config(),
                        'weights': self.layer.get_weights()}, open("tv_layer.pkl", "wb"))
        else:
            self.load()

    def load(self):
        from_disk = pickle.load(open("tv_layer.pkl", "rb"))
        new_v = tf.keras.layers.TextVectorization.from_config(
            from_disk['config'])
        new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        new_v.set_weights(from_disk['weights'])
        self.layer = new_v
        self.vocab = new_v.get_vocabulary()

    def get_data(self, train_ratio=TRAIN_RATIO):

        def prepare_inputs_labels(text):
            text = tf.expand_dims(text, -1)
            tokenized_sentences = self.layer(text)
            x = tokenized_sentences[:, :-1]
            y = tokenized_sentences[:, 1:]
            return (tokenized_sentences, x), y

        self.train_text_ds = tf.data.Dataset.from_tensor_slices(
            self.dataset[:int(len(self.dataset)*train_ratio)])
        self.train_text_ds = self.train_text_ds.batch(BATCH_SIZE)
        self.train_text_ds = self.train_text_ds.map(prepare_inputs_labels)
        self.train_text_ds = self.train_text_ds.prefetch(tf.data.AUTOTUNE)

        self.test_text_ds = tf.data.Dataset.from_tensor_slices(
            self.dataset[int(len(self.dataset)*train_ratio):])
        self.test_text_ds = self.test_text_ds.batch(BATCH_SIZE)
        self.test_text_ds = self.test_text_ds.map(prepare_inputs_labels)
        self.test_text_ds = self.test_text_ds.prefetch(tf.data.AUTOTUNE)

        train = self.train_text_ds
        test = self.test_text_ds
        return train, test


if __name__ == '__main__':
    dataset = Dataset()
    train, test = dataset.get_data()

    for i in train:
        print(i[0], i[1])
        break
