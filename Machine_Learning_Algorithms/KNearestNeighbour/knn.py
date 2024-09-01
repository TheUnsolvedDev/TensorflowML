import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Dataset:
    num_points = 1000
    num_classes = 3

    def __init__(self, plot=False) -> None:
        self.plot = plot

    def create(self):
        mus = [[0, 0], [5, 5], [10, 0]]
        covars = [
            [[1.2, 0], [0, 1.4]],
            [[1.5, 0], [0, 1.5]],
            [[1, 0], [0, 1]]
        ]
        x0 = np.random.multivariate_normal(
            mus[0], covars[0], Dataset.num_points)
        x1 = np.random.multivariate_normal(
            mus[1], covars[1], Dataset.num_points)
        x2 = np.random.multivariate_normal(
            mus[2], covars[2], Dataset.num_points)
        y0 = np.zeros(Dataset.num_points)
        y1 = np.ones(Dataset.num_points)
        y2 = np.ones(Dataset.num_points)*2
        X, y = np.concatenate([x0, x1, x2], axis=0), np.concatenate([y0, y1, y2], axis=0)
        if self.plot:
            self.plot_graph(X, y)
        return X, y

    def plot_graph(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()


class KnearestNeighbors:
    def __init__(self, data, labels, K=10):
        self.data = data
        self.labels = labels
        self.classes = np.unique(self.labels)
        self.K = K

    @tf.function
    def euclidean_distance(self, point1, point2):
        return tf.math.sqrt(tf.reduce_sum(tf.square(point1 - point2), axis=1))

    def predict(self, point):
        distances = self.euclidean_distance(self.data, point)
        k_indices = np.argsort(distances)[:self.K]
        k_class = np.array([self.labels[i] for i in k_indices])
        counts = [np.count_nonzero(k_class == i) for i in self.classes]
        return np.argmax(counts)


if __name__ == '__main__':
    data = Dataset(plot=True)
    X, y = data.create()
    
    knn = KnearestNeighbors(X, y)
    test = [[0, 0], [6, 5], [10, 0]]
    
    print(knn.predict(test[1]))
