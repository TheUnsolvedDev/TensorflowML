import silence_tensorflow.auto
import tensorflow as tf
import random
import string
import pickle
import os

from config import *


def custom_standardization(input_string):
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


class Dataset:
    def __init__(self):
        filenames = []
        # directories = [
        #     "aclImdb/train/pos",
        #     "aclImdb/train/neg",
        #     "aclImdb/test/pos",
        #     "aclImdb/test/neg",
        #     "aclImdb/train/unsup",
        # ]
        directories = [
            # 'AmItheAsshole', 
            # 'offmychest', 
            'DataWrangler/new']
        for dir in directories:
            for f in os.listdir(dir):
                filenames.append(os.path.join(dir, f))

        random.shuffle(filenames)
        text_ds = tf.data.TextLineDataset(filenames)
        text_ds = text_ds.shuffle(buffer_size=256)
        self.text_ds = text_ds.batch(BATCH_SIZE)

        if not os.path.exists("tv_layer.pkl"):
            print("vectorizing data and saving")
            self.vectorize_layer = tf.keras.layers.TextVectorization(
                standardize=custom_standardization,
                max_tokens=VOCAB_SIZE - 1,
                output_mode="int",
                output_sequence_length=MAX_LENGTH + 1,
            )
            self.vectorize_layer.adapt(self.text_ds)
            self.vocab = self.vectorize_layer.get_vocabulary()
            # pickle.dump({'config': self.vectorize_layer.get_config(),
            #             'weights': self.vectorize_layer.get_weights()}, open("tv_layer.pkl", "wb"))
        else:
            print("loading from disk")
            self.load()

    def load(self):
        from_disk = pickle.load(open("tv_layer.pkl", "rb"))
        new_v = tf.keras.layers.TextVectorization.from_config(
            from_disk['config'])
        new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        new_v.set_weights(from_disk['weights'])
        self.vectorize_layer = new_v
        self.vocab = new_v.get_vocabulary()

    def get_data(self):
        def prepare_lm_inputs_labels(text):
            text = tf.expand_dims(text, -1)
            tokenized_sentences = self.vectorize_layer(text)
            x = tokenized_sentences[:, :-1]
            y = tokenized_sentences[:, 1:]
            return x, y
        text_ds = self.text_ds.map(
            prepare_lm_inputs_labels, num_parallel_calls=tf.data.AUTOTUNE)
        text_ds = text_ds.prefetch(tf.data.AUTOTUNE)
        return text_ds
