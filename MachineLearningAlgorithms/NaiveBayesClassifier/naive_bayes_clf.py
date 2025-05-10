import tensorflow as tf
import numpy as np
import tqdm

NUM_DATA = 10_000
NUM_FEATURES = 10
TRAIN_RATIO = 0.8


class Dataset:
    def __init__(self) -> None:
        self.data = tf.random.normal((NUM_DATA, NUM_FEATURES))
        self.weight = tf.ones((NUM_FEATURES, 1))
        self.bias = tf.ones(1)

    def get_data(self):
        indices = tf.random.shuffle(range(NUM_DATA))
        result = tf.matmul(self.data, self.weight) + \
            self.bias + tf.random.normal((NUM_DATA, 1))
        result = tf.cast(tf.math.sign(result) > 0, dtype=tf.float32)
        train_data, train_label = tf.gather(self.data, indices[:int(len(
            result)*TRAIN_RATIO)]), tf.gather(result, indices[:int(len(result)*TRAIN_RATIO)])
        test_data, test_label = tf.gather(self.data, indices[int(len(
            result)*TRAIN_RATIO):]), tf.gather(result, indices[int(len(result)*TRAIN_RATIO):])
        print(len(train_data), len(test_data),train_label.shape)
        return (train_data, train_label), (test_data, test_label)



class NaiveBayes:
    def __init__(self, train_x, train_y):
        train_x = np.array(train_x)
        train_y = np.array(train_y).reshape(-1,)
        self.num_features = len(train_x[0])
        self.classes = np.unique(train_y)
        self.class_count = [np.sum(train_y == i) for i in self.classes]

        self.means = {}
        self.covars = {}
        self.priors = {}

        for idx, c in enumerate(self.classes):
            X_c = train_x[train_y == c]
            self.means[c] = X_c.mean(axis=0)
            self.covars[c] = np.cov(X_c, rowvar=False)
            self.priors[c] = len(X_c)/len(train_x)

    def multivariate_gaussian_pdf(self, x, mean, cov_matrix):
        n = len(mean)
        x_minus_mean = x - mean
        cov_inverse = np.linalg.inv(cov_matrix)
        det_cov = np.linalg.det(cov_matrix)
        coeff = 1.0 / (np.sqrt((2 * np.pi) ** n * det_cov))
        exponent = -0.5 * \
            np.dot(np.dot(x_minus_mean.T, cov_inverse), x_minus_mean)
        return coeff * np.exp(exponent)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            class_scores = {}
            for cls in self.classes:
                mean = self.means[cls]
                cov_matrix = self.covars[cls]
                prior = self.priors[cls]
                pdf = self.multivariate_gaussian_pdf(x, mean, cov_matrix)
                class_scores[cls] = pdf * prior
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return predictions
    
def binary_accuracy(y_true, y_pred):
    y_pred_binary = tf.cast(tf.math.round(y_pred), dtype=tf.float32)
    correct_predictions = tf.equal(y_true, y_pred_binary)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))
    return accuracy.numpy()*100


if __name__ == '__main__':
    obj = Dataset()
    train, test = obj.get_data()

    clf = NaiveBayes(train[0],train[1])
    train_pred = clf.predict(train[0])
    test_pred = clf.predict(test[0])
    
    print('Train Accuracy:',binary_accuracy(train[1],train_pred))
    print('Test Accuracy:',binary_accuracy(test[1],test_pred))
