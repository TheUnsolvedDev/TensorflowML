import tensorflow as tf
import numpy as np
from config import *
from dataset import *
from model import *

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class TextGenerator:
    def __init__(self, model_path="model.keras", max_length=MAX_LEN, temperature=0.9, top_k=10):
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k

        self.strategy = tf.distribute.MirroredStrategy()
        print(f"Using {self.strategy.num_replicas_in_sync} GPUs")

        with self.strategy.scope():
            self.model = tf.keras.models.load_model(model_path, compile=False)

        self.data = Dataset('dataset.txt')
        self.vectorizer = self.data.vectorizer
        self.vocab = self.data.vocab
        self.inverse_vocab = {i: word for i, word in enumerate(self.vocab)}

    def sample_from_top_k(self, logits):
        logits = logits / self.temperature
        values, indices = tf.math.top_k(logits, k=self.top_k)
        values = tf.nn.softmax(values)
        sampled_idx = tf.random.categorical(tf.math.log([values]), num_samples=1)[0, 0]
        return indices[sampled_idx].numpy()

    def generate_text(self, seed_text):
        input_text = seed_text
        for _ in range(self.max_length):
            vec = self.vectorizer([input_text])
            vec = vec[:, -self.max_length:]

            pad_len = self.max_length - vec.shape[1]
            if pad_len > 0:
                padding = tf.zeros((1, pad_len), dtype=tf.int32)
                vec = tf.concat([padding, vec], axis=1)

            preds = self.model(vec, training=False)
            logits = preds[0, -1]
            next_token_id = self.sample_from_top_k(logits)
            next_word = self.inverse_vocab.get(next_token_id, "")

            if next_word == "":
                break
            input_text += " " + next_word
        return input_text


if __name__ == "__main__":
    generator = TextGenerator()

    while True:
        seed = input("\nEnter a prompt (or type 'exit' to quit): ")
        if seed.lower() == "exit":
            break
        output = generator.generate_text(seed)
        print("\n[Generated Text]")
        print(output)
