import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import tqdm

from config import *


class Dataset:
    def __init__(self) -> None:
        self.num_classes = NUM_CLASSES
        self.num_features = NUM_FEATURES
        self.num_samples = NUM_SAMPLES

        means = np.array([3*i*np.ones(self.num_features)
                         for i in range(self.num_classes)])

        self.data = np.concatenate([np.random.multivariate_normal(
            mean, 0.1*np.eye(self.num_features), self.num_samples//self.num_classes) for mean in means])
        self.labels = np.concatenate([np.full(self.num_samples//self.num_classes, i)
                                      for i in range(self.num_classes)])

    def shuffle(self, data, labels):
        indices = np.random.permutation(len(data))
        return data[indices], labels[indices]

    def dataset(self, train_test_ratio=TRAIN_TEST_RATIO):
        self.data,self.labels = self.shuffle(self.data, self.labels)
        train_data = self.data[:int(len(self.data) * train_test_ratio)]
        train_labels = self.labels[:int(len(self.labels) * train_test_ratio)]
        test_data = self.data[int(len(self.data) * train_test_ratio):]
        test_labels = self.labels[int(len(self.labels) * train_test_ratio):]
        return train_data, train_labels, test_data, test_labels


class NaiveBayesClassifier:
    def __init__(self, num_classes, num_features, num_samples):
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_samples = num_samples

    def fit(self, X, y):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.int32)
        self.classes, _, counts = tf.unique_with_counts(y)
        self.priors = tf.cast(counts,tf.float32) / tf.cast(tf.shape(y)[0], tf.float32)
        self.means = tf.TensorArray(dtype=tf.float32, size=len(self.classes))
        self.vars = tf.TensorArray(dtype=tf.float32, size=len(self.classes))

        for idx, c in enumerate(self.classes):
            X_c = tf.boolean_mask(X, tf.equal(y, c))
            self.means = self.means.write(idx, tf.reduce_mean(X_c, axis=0))
            self.vars = self.vars.write(idx, tf.math.reduce_variance(X_c, axis=0))
        
        self.means = self.means.stack()
        self.vars = self.vars.stack()

    def _calculate_likelihood(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        eps = 1e-6  # to avoid division by zero
        coeff = 1.0 / tf.sqrt(2.0 * np.pi * var + eps)
        exponent = tf.exp(-((x - mean) ** 2) / (2.0 * var + eps))
        return coeff * exponent

    def _calculate_posterior(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = tf.math.log(self.priors[idx])
            likelihood = tf.reduce_sum(tf.math.log(self._calculate_likelihood(idx, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[tf.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._calculate_posterior(x) for x in X]
        return tf.convert_to_tensor(y_pred)


if __name__ == '__main__':
    dataset = Dataset()
    train_data, train_labels, test_data, test_labels = dataset.dataset()
    classifier = NaiveBayesClassifier(NUM_CLASSES, NUM_FEATURES, NUM_SAMPLES)
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)
    print(predictions,test_labels)
    accuracy = np.mean(predictions == test_labels)
    print('The accuracy of the model is', accuracy)
