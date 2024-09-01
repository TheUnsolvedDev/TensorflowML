import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

NUM_DATA = 500
NUM_FEATURES = 2
TRAIN_RATIO = 0.8

# def generateBatchBipolar(n, mu=0.5, sigma=0.2):
#     """ Two gaussian clouds on each side of the origin """
#     X = np.random.normal(mu, sigma, (n, 2))
#     yB = np.random.uniform(0, 1, n) > 0.5
#     # y is in {-1, 1}
#     y = 2. * yB - 1
#     X *= y[:, np.newaxis]
#     X -= X.mean(axis=0)
#     return X, y


class Dataset:
    def __init__(self):
        pass

    def get_data_linear(self):
        X = np.random.normal(0.5, 0.2, (NUM_DATA, 2))
        yB = np.random.uniform(0, 1, NUM_DATA) > 0.5
        y = 2. * yB - 1
        X *= y[:, np.newaxis]
        X -= X.mean(axis=0)
        train_data, train_label = X[:int(
            len(X)*TRAIN_RATIO)], y[:int(len(X)*TRAIN_RATIO)]
        test_data, test_label = X[int(
            len(X)*TRAIN_RATIO):], y[int(len(X)*TRAIN_RATIO):]

        return (train_data,train_label),(test_data,test_label)

    def get_data_non_linear(self):
        X = np.random.normal(0.5, 0.2, (NUM_DATA, 2))
        yB0 = np.random.uniform(0, 1, NUM_DATA) > 0.5
        yB1 = np.random.uniform(0, 1, NUM_DATA) > 0.5
        y0 = 2. * yB0 - 1
        y1 = 2. * yB1 - 1
        X[:, 0] *= y0
        X[:, 1] *= y1
        X -= X.mean(axis=0)
        y = y0*y1
        train_data, train_label = X[:int(
            len(X)*TRAIN_RATIO)], y[:int(len(X)*TRAIN_RATIO)]
        test_data, test_label = X[int(
            len(X)*TRAIN_RATIO):], y[int(len(X)*TRAIN_RATIO):]

        return (train_data,train_label),(test_data,test_label)


if __name__ == '__main__':
    obj = Dataset()
    obj.get_data()
