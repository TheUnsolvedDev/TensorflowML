import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *


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

    def dataset(self, train_test_ratio=TRAIN_TEST_RATIO):
        train_data = int(len(self.data) * train_test_ratio)
        return self.data[:train_data], self.data[train_data:], self.labels[:train_data], self.labels[train_data:]


class LinearRegression:
    def __init__(self) -> None:
        self.random_weights = np.random.normal(
            0, 1, (NUM_FEATURES, 1)).astype(np.float32)
        self.random_weights = tf.Variable(self.random_weights)
        self.random_bias = np.random.normal(0, 1, (1,)).astype(np.float32)
        self.random_bias = tf.Variable(self.random_bias)
        self.batch_size = BATCH_SIZE
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    @tf.function
    def r2_score(self, predictions, labels):
        return 1 - tf.reduce_sum(tf.square(predictions - labels)) / tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels)))

    @tf.function
    def mean_squared_loss(self, predictions, labels):
        return tf.reduce_mean(tf.square(predictions - labels))

    @tf.function
    def train_one_step(self, data, labels):
        with tf.GradientTape() as tape:
            predictions = tf.matmul(
                data, self.random_weights) + self.random_bias
            loss = self.mean_squared_loss(predictions, labels)

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

            predictions = tf.matmul(
                batch_data, self.random_weights) + self.random_bias
            r2 = self.r2_score(predictions, batch_labels).numpy().astype(np.float32)
            loss = self.mean_squared_loss(predictions, batch_labels).numpy().astype(np.float32)
            weights = self.random_weights.numpy().reshape(-1,)
            bias = self.random_bias.numpy().reshape(-1,)
            print(
                f"Epoch: {epoch}, Loss: {loss}, R2: {r2}", end="\r", flush=True)

    def predict(self, data):
        return tf.matmul(data, self.random_weights) + self.random_bias


if __name__ == '__main__':
    data = Dataset()
    print('Weights:', data.weights, 'Bias:', data.bias)
    model = LinearRegression()

    train_data, test_data, train_labels, test_labels = data.dataset()
    model.fit(train_data, train_labels)
