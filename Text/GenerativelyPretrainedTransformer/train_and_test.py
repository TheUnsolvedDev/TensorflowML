import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *
from model import GPTModel
from dataset import Dataset
import tqdm

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class SentenceCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("Sentence Generated:", end=" ")
        sentence = 'I am'

        for _ in range(MAX_LENGTH-1):
            tokenize = tf.expand_dims(text_layer(sentence)[:-1], axis=0)
            next_one = self.model.predict(tokenize, verbose=False)[0]
            index = len(sentence.split(" ")) + 1
            next_word = np.argmax(next_one[index])
            word = vocab[next_word]
            sentence += " " + word
            print(sentence)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_GPT.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
    SentenceCallback(),
]


def word_accuracy(y_true, y_pred):
    y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.int32)
    y_true = tf.cast(y_true, dtype=tf.int32)
    return tf.reduce_mean(tf.cast(tf.math.equal(y_true, y_pred), dtype=tf.float32))


def sentence_accuracy(y_true, y_pred):
    y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.int32)
    y_true = tf.cast(y_true, dtype=tf.int32)
    return tf.reduce_all(tf.math.equal(y_true, y_pred), axis=-1)


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, mask_value: int = 0, reduction: str = 'none') -> None:
        super(MaskedLoss, self).__init__()
        self.mask_value = mask_value
        self.reduction = reduction
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=reduction)

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        mask = y_true != self.mask_value
        loss = self.loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss


class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, mask_value: int = 0, name: str = 'masked_accuracy') -> None:
        super(MaskedAccuracy, self).__init__(name=name)
        self.mask_value = mask_value
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    @tf.function
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        pred = tf.argmax(y_pred, axis=2)
        label = tf.cast(y_true, pred.dtype)
        match = label == pred

        mask = label != self.mask_value
        match = match & mask
        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        match = tf.reduce_sum(match)
        mask = tf.reduce_sum(mask)

        self.total.assign_add(match)
        self.count.assign_add(mask)

    def result(self) -> tf.Tensor:
        return self.total / self.count


class CERMetric(tf.keras.metrics.Metric):
    def __init__(self, end_token, padding_token: int = 0, name="CER", **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = tf.Variable(
            0.0, name="cer_accumulator", dtype=tf.float32)
        self.batch_counter = tf.Variable(
            0, name="batch_counter", dtype=tf.int32)

        self.padding_token = padding_token
        self.end_token = end_token

    def get_cer(self, pred, y_true, padding=-1):
        equal = tf.equal(pred, self.end_token)
        equal_int = tf.cast(equal, tf.int64)
        end_token_index = tf.argmax(equal_int, axis=1)
        new_range = tf.range(tf.shape(pred)[1], dtype=tf.int64)
        range_matrix = tf.tile(new_range[None, :], [tf.shape(pred)[0], 1])
        mask = range_matrix <= tf.expand_dims(end_token_index, axis=1)
        masked_pred = tf.where(mask, pred, padding)
        sparse_pred = tf.RaggedTensor.from_tensor(
            masked_pred, padding=padding).to_sparse()
        sparse_true = tf.RaggedTensor.from_tensor(
            y_true, padding=padding).to_sparse()
        distance = tf.edit_distance(sparse_pred, sparse_true, normalize=True)
        return distance

    # @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        pred = tf.argmax(y_pred, axis=2)
        distance = self.get_cer(pred, y_true, self.padding_token)
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.batch_counter.assign_add(len(y_true))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, tf.cast(self.batch_counter, tf.float32))


def train_and_test_model():
    global text_layer, vocab
    model = GPTModel()
    model.summary(expand_nested=True)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=[MaskedLoss()],
        metrics=[MaskedAccuracy()]
    )

    dataset = Dataset()
    train_dataset, test_dataset = dataset.get_data()
    vocab = dataset.vocab
    text_layer = dataset.layer
    model.fit(train_dataset, epochs=EPOCHS,
              validation_data=test_dataset, callbacks=callbacks)


if __name__ == '__main__':
    train_and_test_model()
    print(text_layer('I am hid on the way'))
