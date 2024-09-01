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


class KNearestNeighbors:
    def __init__(self, num_classes, num_features):
        self.num_features = num_features
        self.k = num_classes

    def most_common(self, tensor):
        unique, _, counts = tf.unique_with_counts(tensor)
        max_count_index = tf.argmax(counts)
        most_common_element = unique[max_count_index]
        return most_common_element

    def euclidean_distance(self, x1, x2):
        return tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), axis=1))

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def predict(self, data_point):
        distances = self.euclidean_distance(self.data, data_point)
        sorted_indices = tf.argsort(distances)[:self.k]
        k_nearest_labels = tf.gather(self.labels, sorted_indices)
        return self.most_common(k_nearest_labels)

    def accuracy_score(self, labels, predictions):
        correct = tf.reduce_sum(tf.cast(labels == predictions, tf.int32))
        return correct / len(labels)

    def evaluate(self, data, labels):
        predictions = [self.predict(data_point) for data_point in tqdm.tqdm(data)]
        predictions = tf.stack(predictions)
        return self.accuracy_score(labels, predictions)


if __name__ == "__main__":
    data = Dataset()
    train_data, train_labels, test_data, test_labels = data.dataset()
    
    model = KNearestNeighbors(NUM_CLASSES, NUM_FEATURES, )
    model.fit(train_data, train_labels)
    print('The accuracy of the model is',model.evaluate(test_data, test_labels))
    
    # print(model.predict(test_data, test_labels, test_data[0]))
