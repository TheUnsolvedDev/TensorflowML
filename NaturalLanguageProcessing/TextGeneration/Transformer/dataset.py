import tensorflow as tf
import os
from config import VOCAB_SIZE, MAX_LEN, BATCH_SIZE

# Fallback defaults
VOCAB_SIZE = VOCAB_SIZE if 'VOCAB_SIZE' in locals() else 20000
MAX_LEN = MAX_LEN if 'MAX_LEN' in locals() else 128
BATCH_SIZE = BATCH_SIZE if 'BATCH_SIZE' in locals() else 64


def custom_standardization(input_text):
    lowercase = tf.strings.lower(input_text)
    cleaned = tf.strings.regex_replace(lowercase, r"[^\w\s]", "")
    return cleaned


class Dataset:
    def __init__(self, data_path='dataset.txt', vocab_path='vocab.txt'):
        self.data_path = data_path
        self.vocab_path = vocab_path

        self.vectorizer, self.vocab = self._build_or_load_vectorizer()

    def _build_or_load_vectorizer(self):
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, "r", encoding='utf-8') as f:
                vocab = [line.strip() for line in f.readlines()]
            print(f"[INFO] Loaded vocabulary from {self.vocab_path}")
        else:
            print(f"[INFO] Building vocabulary from {self.data_path}...")
            raw_text_ds = tf.data.TextLineDataset(self.data_path) \
                .filter(lambda x: tf.strings.length(x) > 0) \
                .map(custom_standardization, num_parallel_calls=tf.data.AUTOTUNE) \
                .batch(32) \
                .prefetch(tf.data.AUTOTUNE)

            vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=VOCAB_SIZE,
                output_mode="int",
                output_sequence_length=MAX_LEN+1,
                standardize=None
            )
            vectorizer.adapt(raw_text_ds)
            vocab = vectorizer.get_vocabulary()

            with open(self.vocab_path, "w", encoding='utf-8') as f:
                for word in vocab:
                    f.write(word + "\n")
            print(f"[INFO] Saved vocabulary to {self.vocab_path}")

        # Rebuild vectorizer from saved vocab
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=len(vocab),
            output_mode="int",
            output_sequence_length=MAX_LEN+1,
            standardize=custom_standardization
        )
        vectorizer.set_vocabulary(vocab)
        return vectorizer, vocab

    def get_dataset(self, for_training=True):
        raw_ds = tf.data.TextLineDataset(self.data_path) \
            .filter(lambda x: tf.strings.length(x) > 0) \
            .map(lambda x: self.vectorizer(x), num_parallel_calls=tf.data.AUTOTUNE)

        def split_input_target(sequence):
            return sequence[:-1], sequence[1:]

        ds = raw_ds.map(split_input_target,
                        num_parallel_calls=tf.data.AUTOTUNE)
        if for_training:
            ds = ds.shuffle(10000).repeat()
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    def text_to_vector(self, texts):
        return self.vectorizer(tf.constant(texts))

    def vector_to_text(self, vectors):
        inverse_vocab = {i: word for i, word in enumerate(self.vocab)}
        result = []
        for vec in vectors.numpy():
            result.append(" ".join([inverse_vocab.get(i, "")
                          for i in vec if i > 0]))
        return result


if __name__ == '__main__':
    dataset = Dataset(data_path='dataset.txt')

    # Load dataset
    train_ds = dataset.get_dataset()
    for x, y in train_ds.take(1):
        print("Input IDs:", x.numpy().shape)
        print("Target IDs:", y.numpy().shape)

    # Test vectorization
    test_texts = ["Audit meeting schedule!", "once upon a time in a big forest there lived a rhinoceros named roxy roxy loved to climb she climbed trees rocks and hills one day roxy found an icy hill she had never seen anything like it before it was shiny and cold and she wanted to climb itroxy tried to climb the icy hill but it was very slippery she tried again and again but she kept falling down roxy was sad she wanted to climb the icy hill so much then she saw a little bird named billy billy saw that roxy was sad and asked why are you sad roxyroxy told billy about the icy hill and how she couldnt climb it billy said i have an idea lets find some big leaves to put under your feet they will help you climb the icy hill roxy and billy looked for big leaves and found some roxy put the leaves under her feet and tried to climb the icy hill againthis time roxy didnt slip she climbed and climbed until she reached the top of the icy hill roxy was so happy she and billy played on the icy hill all day from that day on roxy and billy were the best of friends and they climbed and played together all the time and roxy learned that with a little help from a friend she could climb anything"]
    vectors = dataset.text_to_vector(test_texts)
    print("\nVectorized:\n", vectors.numpy())

    # Test de-vectorization
    recovered = dataset.vector_to_text(vectors)
    print("\nRecovered text:\n", recovered)
