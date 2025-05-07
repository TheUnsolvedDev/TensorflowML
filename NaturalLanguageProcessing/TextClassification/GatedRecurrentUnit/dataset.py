import tensorflow as tf
import numpy as np
import pandas as pd
import os

from config import *  # ensure this defines MAX_LEN or default it below

# ========================
# Custom standardization
# ========================


def custom_standardization(input_text):
    lowercase = tf.strings.lower(input_text)
    cleaned = tf.strings.regex_replace(lowercase, r"[^\w\s]", "")
    return cleaned

# ========================
# Dataset class
# ========================


class Dataset:
    def __init__(self, data_path='dataset/enron_data_fraud_labeled.csv', vocab_path='vocab.txt'):
        df = pd.read_csv(data_path, low_memory=False)
        self.data, self.label = df['Body'], df['Label']
        self.vocab_path = vocab_path

        # Build or load vectorizer
        self.vectorizer, self.vocab = self._build_or_load_vectorizer(self.data)

    def _build_or_load_vectorizer(self, text_data):
        """
        Loads vectorizer from saved vocab or creates and saves it if missing.
        """
        if os.path.exists(self.vocab_path):
            # Load existing vocab
            with open(self.vocab_path, "r") as f:
                vocab = [line.strip() for line in f.readlines()]
            print(f"[INFO] Loaded vocabulary from {self.vocab_path}")
        else:
            # Create and adapt vectorizer
            raw_text_ds = tf.data.Dataset.from_tensor_slices(
                text_data).batch(32)
            vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=VOCAB_SIZE,
                output_mode="int",
                output_sequence_length=MAX_LEN,
                standardize=custom_standardization
            )
            vectorizer.adapt(raw_text_ds)
            vocab = vectorizer.get_vocabulary()
            # Save vocabulary
            with open(self.vocab_path, "w") as f:
                for word in vocab:
                    f.write(word + "\n")
            print(f"[INFO] Saved vocabulary to {self.vocab_path}")

        # Rebuild vectorizer from vocab
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=len(vocab),
            output_mode="int",
            output_sequence_length=MAX_LEN,
            standardize=custom_standardization
        )
        vectorizer.set_vocabulary(vocab)
        return vectorizer, vocab

    def text_to_vector(self, texts):
        text_ds = tf.constant(texts)
        return self.vectorizer(text_ds)

    def vector_to_text(self, vectors):
        inverse_vocab = {i: word for i, word in enumerate(self.vocab)}
        texts = []
        for vec in vectors.numpy():
            words = [inverse_vocab.get(i, "") for i in vec if i > 0]
            texts.append(" ".join(words))
        return texts

    def prepare(self, ds, batch_size=BATCH_SIZE):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def get_data(self):
        X = self.text_to_vector(self.data)
        y = tf.convert_to_tensor(self.label.tolist())

        full_ds = tf.data.Dataset.from_tensor_slices((X, y))
        full_ds = full_ds.shuffle(
            buffer_size=100_000, reshuffle_each_iteration=False)

        total = len(X)
        val_size = int(0.1 * total)
        train_size = total - val_size

        train_ds = full_ds.take(train_size)
        val_ds = full_ds.skip(train_size)
        train_ds = self.prepare(train_ds)
        val_ds = self.prepare(val_ds)
        return train_ds, val_ds


# ========================
# Example usage
# ========================
if __name__ == '__main__':
    dataset = Dataset()

    example_texts = [
        "Audit Committee Materials meeting!!!",
        "The WTI Bullet swap contracts $$$"
    ]

    vecs = dataset.text_to_vector(example_texts)
    print("Vectors:\n", vecs.numpy())

    recovered = dataset.vector_to_text(vecs)
    print("\nRecovered:\n", recovered)
