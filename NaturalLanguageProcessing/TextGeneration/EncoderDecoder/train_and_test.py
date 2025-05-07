import tensorflow as tf
import numpy as np
import os
import time
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
    def __init__(self, model, encoder_model, decoder_model, vectorizer, vocab,
                 seed_texts=None, max_length=MAX_LEN, temperature=0.9, top_k=10):
        super().__init__()
        self.model = model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.vectorizer = vectorizer
        self.vocab = vocab
        self.inverse_vocab = {i: word for i, word in enumerate(self.vocab)}
        self.max_length = max_length
        self.seed_texts = seed_texts if seed_texts else [
            "Once upon a time", "The company will", "She decided to"]
        self.temperature = temperature
        self.top_k = top_k

    def sample_from_top_k(self, logits):
        logits = logits / self.temperature
        values, indices = tf.math.top_k(
            logits, k=min(self.top_k, logits.shape[-1]))
        values = tf.nn.softmax(values)
        sampled_idx = tf.random.categorical(
            tf.math.log([values]), num_samples=1)[0, 0]
        return indices[sampled_idx].numpy()

    def generate_text(self, seed_text):
        input_seq = Dataset.text_to_vector(seed_text, self.vectorizer)

        # Get encoder states
        encoder_states = self.encoder_model.predict(input_seq, verbose=0)

        # Initialize target sequence with start token
        target_seq = np.zeros((1, 1), dtype=np.int64)
        target_seq[0, 0] = 0  # Start token

        generated_tokens = []

        for _ in range(self.max_length):
            # Predict next token and states
            outputs = self.decoder_model.predict(
                [target_seq] + encoder_states, verbose=0)
            output_tokens = outputs[0]  # First element is token probabilities
            encoder_states = outputs[1:]  # Rest are the states

            # Sample a token
            sampled_token_index = self.sample_from_top_k(
                output_tokens[0, -1, :])
            generated_tokens.append(sampled_token_index)

            # Exit if end of sequence or padding token
            if sampled_token_index == 0:
                break

            # Update target sequence
            target_seq = np.zeros((1, 1), dtype=np.int64)
            target_seq[0, 0] = sampled_token_index

        # Convert token IDs to text
        generated_text = " ".join([self.inverse_vocab.get(
            i, "") for i in generated_tokens if i > 0])

        return generated_text

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n----- Generating text after epoch {epoch+1} -----")
        for seed_text in self.seed_texts:
            start = time.time()
            generated_text = self.generate_text(seed_text)
            print(f"Seed: {seed_text}")
            print(f"Generated: {generated_text}")
            print(f"Generation time: {time.time() - start:.2f}s\n")

    def on_epoch_end(self, epoch, logs=None):
        print("\n[Sampled text]")
        print(self.generate_text(self.seed_text))
        print("")

# 7. Main Training Script


def main():
    data = Dataset('dataset.txt')
    train_ds = data.get_dataset()
    vectorizer, vocab = data.vectorizer, data.vocab
    num_train_examples = 100  # 2141479

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())
    print(f"[INFO] Number of devices: {strategy.num_replicas_in_sync}")

    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)

    with strategy.scope():
        model = build_seq2seq_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False, reduction='none'),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(),
            ]
        )

    model.summary()
    encoder_inference = create_inference_encoder(model)
    decoder_inference = create_inference_decoder(model)

    model.fit(
        train_dist_ds,
        epochs=EPOCHS,
        steps_per_epoch=num_train_examples // BATCH_SIZE,
        callbacks=[
            TextGenerationCallback(
                model, encoder_inference, decoder_inference, vectorizer, vocab, seed_texts),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='model.keras',
                save_best_only=True,
                monitor='loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=3
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
    )

    # Save the model
    # model.save('seq2seq_final_model.h5')

    print("Training completed. Model saved as 'seq2seq_final_model.h5'")
    seed_texts = ["Audit meeting schedule", "once upon a time",
                  "the company will", "she wanted to"]
    # Test the model with some examples
    print("\n----- Final text generation examples -----")
    callback = TextGenerationCallback(
        model, encoder_inference, decoder_inference, vectorizer, vocab, seed_texts)
    for seed in seed_texts:
        generated = callback.generate_text(seed)
        print(f"\nSeed: {seed}")
        print(f"Generated: {generated}")


if __name__ == '__main__':
    main()
