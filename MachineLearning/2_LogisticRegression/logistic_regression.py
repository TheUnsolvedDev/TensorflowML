import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Dataset:
    def __init__(self) -> None:
        self.weights = np.random.normal(
            0, 1, (NUM_FEATURES, 1)).astype(np.float32)
        self.bias = np.random.normal(0, 1, (1,)).astype(np.float32)

        self.data = np.random.normal(0, 1, (NUM_SAMPLES, NUM_FEATURES))
        self.data = (self.data - np.mean(self.data, axis=0)) / \
            np.std(self.data, axis=0)
        self.labels = np.dot(self.data, self.weights) + self.bias
        self.labels += np.random.normal(0, 0.01, (NUM_SAMPLES, 1))
        self.labels = np.where(sigmoid(self.labels) > 0.5, 1, 0)

    def dataset(self, train_test_ratio=TRAIN_TEST_RATIO):
        train_data = int(len(self.data) * train_test_ratio)
        return self.data[:train_data], self.data[train_data:], self.labels[:train_data], self.labels[train_data:]


class LogisticRegression:
    def __init__(self) -> None:
        self.random_weights = np.random.normal(
            0, 1, (NUM_FEATURES, 1)).astype(np.float32)
        self.random_weights = tf.Variable(self.random_weights)
        self.random_bias = np.random.normal(0, 1, (1,)).astype(np.float32)
        self.random_bias = tf.Variable(self.random_bias)
        self.batch_size = BATCH_SIZE
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    def binary_accuracy(self, predictions, labels):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(predictions), labels), tf.float32))

    def binary_crossentropy(self, predictions, labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=predictions))

    @tf.function
    def train_one_step(self, data, labels):
        with tf.GradientTape() as tape:
            predictions = tf.nn.sigmoid(
                tf.matmul(data, self.random_weights) + self.random_bias)
            loss = self.binary_crossentropy(predictions, labels)

        gradients = tape.gradient(
            loss, [self.random_weights, self.random_bias])
        return gradients

    def fit(self, data, labels, learning_rate=LEARNING_RATE, epochs=EPOCHS):
        batch_indices = np.arange(len(data))
        np.random.shuffle(batch_indices)

        for epoch in range(epochs):
            for i in range(0, len(data), self.batch_size):
                batch_data = data[batch_indices[i:i +
                                                self.batch_size]].astype(np.float32)
                batch_labels = labels[batch_indices[i:i +
                                                    self.batch_size]].astype(np.float32)
                gradients = self.train_one_step(batch_data, batch_labels)
                self.optimizer.apply_gradients(
                    zip(gradients, [self.random_weights, self.random_bias]))

            predictions = tf.nn.sigmoid(
                tf.matmul(batch_data, self.random_weights) + self.random_bias)
            accuracy = self.binary_accuracy(predictions, batch_labels).numpy().astype(np.float32)
            loss = self.binary_crossentropy(
                predictions, batch_labels).numpy().astype(np.float32)
            weights = self.random_weights.numpy().reshape(-1,)
            bias = self.random_bias.numpy().reshape(-1,)
            print(
                f"Epoch: {epoch}, Loss: {loss}, Acc: {accuracy}", end="\r", flush=True)


if __name__ == '__main__':
    data = Dataset()
    train_data, test_data, train_labels, test_labels = data.dataset()
    model = LogisticRegression()
    
    model.fit(train_data, train_labels)
    