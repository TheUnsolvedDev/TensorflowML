import tensorflow as tf
import numpy as np
import os
from config import *
from dataset import *
from model import *

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



class TextGenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 vectorizer,
                 vocab,
                 max_length=MAX_LEN,
                 seed_text="Once upon a time",
                 temperature=0.9,
                 top_k=10,
                 log_every_n_batches=100,
                 log_file="predictions.txt"):
        super().__init__()
        self.vectorizer = vectorizer
        self.vocab = vocab
        self.inverse_vocab = {i: word for i, word in enumerate(self.vocab)}
        self.max_length = max_length
        self.seed_text = seed_text
        self.temperature = temperature
        self.top_k = top_k

        self.log_every_n = log_every_n_batches
        self.log_file = log_file
        # clear out old file at start of training
        open(self.log_file, "w").close()

    def sample_from_top_k(self, logits):
        logits = logits / self.temperature
        values, indices = tf.math.top_k(logits, k=self.top_k)
        values = tf.nn.softmax(values)
        sampled_idx = tf.random.categorical(tf.math.log([values]), num_samples=1)[0, 0]
        return indices[sampled_idx].numpy()

    def generate_text(self, seed):
        input_text = seed
        for _ in range(self.max_length):
            vec = self.vectorizer([input_text])
            vec = vec[:, -self.max_length:]
            pad_len = self.max_length - vec.shape[1]
            if pad_len > 0:
                padding = tf.zeros((1, pad_len), dtype=tf.int64)
                vec = tf.concat([padding, vec], axis=1)

            preds = self.model(vec, training=False)
            logits = preds[0, -1]
            next_token_id = self.sample_from_top_k(logits)
            next_word = self.inverse_vocab.get(next_token_id, "")
            if not next_word:
                break
            input_text += " " + next_word
        return input_text

    def on_train_batch_end(self, batch, logs=None):
        # every N batches...
        if batch % self.log_every_n == 0:
            pred = self.generate_text(self.seed_text)
            message = f"[batch {batch}]\n{pred}\n\n"
            # append to file
            with open(self.log_file, "a") as f:
                f.write(message)
            # also print to console
            print(f"\n[batch {batch}] {pred}\n")


# 7. Main Training Script
def main():
    data = Dataset('dataset.txt')
    train_ds = data.get_dataset()
    num_train_examples = 2141479

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())
    print(f"[INFO] Number of devices: {strategy.num_replicas_in_sync}")

    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)

    with strategy.scope():
        model = Transformer_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    model.summary()

    model.fit(
        train_dist_ds,
        steps_per_epoch=num_train_examples // BATCH_SIZE,
        epochs=10,
        callbacks=[
            TextGenerationCallback(data.vectorizer, data.vocab),
            tf.keras.callbacks.ModelCheckpoint("model.keras", monitor="loss", save_best_only=True, mode="min"),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2, mode="min")
        ]
    )


if __name__ == '__main__':
    main()