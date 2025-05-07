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

# 1. Perplexity Metric


class Perplexity(tf.keras.metrics.Metric):
    def __init__(self, name="perplexity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(
            name="total_loss", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute loss for the batch
        loss = self.loss_fn(y_true, y_pred)  # shape: (batch, seq_len)
        batch_loss = tf.reduce_mean(loss)
        self.total_loss.assign_add(batch_loss)
        self.count.assign_add(1.0)

    def result(self):
        avg_loss = self.total_loss / self.count
        return tf.exp(avg_loss)

    def reset_states(self):
        self.total_loss.assign(0.0)
        self.count.assign(0.0)

# 2. BLEU-1 Metric (Unigram Overlap)


class BLEU1(tf.keras.metrics.Metric):
    def __init__(self, name="bleu1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_bleu = self.add_weight(
            name="total_bleu", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_ids = tf.argmax(y_pred, axis=-1)

        def compute_bleu(ref, pred):
            ref = [int(x) for x in ref if x != 0]
            pred = [int(x) for x in pred if x != 0]
            ref_counts = {}
            for token in ref:
                ref_counts[token] = ref_counts.get(token, 0) + 1
            pred_counts = {}
            for token in pred:
                pred_counts[token] = pred_counts.get(token, 0) + 1
            match = 0
            for token, count in pred_counts.items():
                match += min(count, ref_counts.get(token, 0))
            total = len(pred)
            return match / total if total > 0 else 0.0

        def batch_bleu(ref_batch, pred_batch):
            scores = [compute_bleu(ref, pred) for ref, pred in zip(ref_batch, pred_batch)]
            return np.array(np.mean(scores), dtype=np.float32)

        bleu_val = tf.py_function(func=batch_bleu, inp=[y_true, y_pred_ids], Tout=tf.float32)
        bleu_val.set_shape([])  # ✅ Required fix
        self.total_bleu.assign_add(bleu_val)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_bleu / self.count

    def reset_states(self):
        self.total_bleu.assign(0.0)
        self.count.assign(0.0)


# 3. Distinct-1 Metric (Unigram Diversity)


class Distinct1(tf.keras.metrics.Metric):
    def __init__(self, name="distinct1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_distinct = self.add_weight(
            name="total_distinct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_ids = tf.argmax(y_pred, axis=-1)

        def compute_distinct(pred):
            pred = [int(x) for x in pred if x != 0]
            distinct_tokens = set(pred)
            return len(distinct_tokens) / len(pred) if len(pred) > 0 else 0.0

        def batch_distinct(pred_batch):
            scores = [compute_distinct(pred) for pred in pred_batch]
            return np.mean(scores).astype(np.float32)

        distinct_val = tf.py_function(func=batch_distinct, inp=[y_pred_ids], Tout=tf.float32)
        distinct_val.set_shape([])  # ✅ Ensure the shape is known
        self.total_distinct.assign_add(distinct_val)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_distinct / self.count

    def reset_states(self):
        self.total_distinct.assign(0.0)
        self.count.assign(0.0)


# 4. Distinct-2 Metric (Bigram Diversity)


class Distinct2(tf.keras.metrics.Metric):
    def __init__(self, name="distinct2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_distinct = self.add_weight(
            name="total_distinct", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_ids = tf.argmax(y_pred, axis=-1)

        def compute_distinct(pred):
            pred = [int(x) for x in pred if x != 0]
            distinct_tokens = set(pred)
            return len(distinct_tokens) / len(pred) if len(pred) > 0 else 0.0

        def batch_distinct(pred_batch):
            scores = [compute_distinct(pred) for pred in pred_batch]
            return np.mean(scores).astype(np.float32)

        distinct2_val = tf.py_function(func=batch_distinct, inp=[y_pred_ids], Tout=tf.float32)
        distinct2_val.set_shape([])  # ✅ Fix: Set shape explicitly
        self.total_distinct.assign_add(distinct2_val)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_distinct / self.count

    def reset_states(self):
        self.total_distinct.assign(0.0)
        self.count.assign(0.0)


# 5. Repetition Rate Metric: 1 - (# unique tokens / total tokens)


class RepetitionRate(tf.keras.metrics.Metric):
    def __init__(self, name="repetition_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_ids = tf.argmax(y_pred, axis=-1)  # shape: (batch, seq_len)

        def compute_repetition(tokens):
            tokens = [int(x) for x in tokens if x != 0]
            if not tokens:
                return 0.0
            return 1.0 - float(len(set(tokens))) / len(tokens)

        def batch_repetition(preds):
            reps = [compute_repetition(seq) for seq in preds]
            return np.mean(reps).astype(np.float32)

        rep_val = tf.py_function(func=batch_repetition, inp=[y_pred_ids], Tout=tf.float32)
        rep_val.set_shape([])  # ✅ Fix: Set shape explicitly
        self.total.assign_add(rep_val)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)



class TextGenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self, vectorizer, vocab, max_length=MAX_LEN, seed_text="Once upon a time", temperature=0.9, top_k=10):
        super().__init__()
        self.vectorizer = vectorizer
        self.vocab = vocab
        self.inverse_vocab = {i: word for i, word in enumerate(self.vocab)}
        self.max_length = max_length
        self.seed_text = seed_text
        self.temperature = temperature
        self.top_k = top_k

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

            if vec.shape[0] == 0 or vec.shape[1] == 0:
                print("Empty vectorized input — skipping generation")
                break

            preds = self.model(vec, training=False)
            logits = preds[0, -1]
            next_token_id = self.sample_from_top_k(logits)
            next_word = self.inverse_vocab.get(next_token_id, "")

            if next_word == "":
                break
            input_text += " " + next_word
        return input_text

    def on_epoch_end(self, epoch, logs=None):
        print("\n[Sampled text]")
        print(self.generate_text(self.seed_text))
        print("")


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
        model = GPT1_Model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none'),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(),
                # Perplexity(),
                # BLEU1(),
                # Distinct1(),
                # Distinct2(),
                # RepetitionRate()
            ]
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