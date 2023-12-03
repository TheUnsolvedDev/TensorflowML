import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import style
style.use('ggplot')

NUM_DATA = 1000
NUM_FEATURES = 2
TRAIN_RATIO = 0.8


class Dataset:
    def __init__(self):
        pass

    def get_data(self):
        X = np.random.normal(0.5, 0.2, (NUM_DATA, 2)).astype(np.float32)
        yB = (np.random.uniform(0, 1, NUM_DATA) > 0.5).astype(np.float32)
        y = (2. * yB - 1)
        X *= y[:, np.newaxis]
        X -= X.mean(axis=0)
        train_data, train_label = X[:int(
            len(X)*TRAIN_RATIO)], y[:int(len(X)*TRAIN_RATIO)]
        test_data, test_label = X[int(
            len(X)*TRAIN_RATIO):], y[int(len(X)*TRAIN_RATIO):]

        return (train_data, train_label), (test_data, test_label)


def support_vector_model():
    inputs = tf.keras.layers.Input(shape=(NUM_FEATURES,))
    outputs = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def accuracy(y_true, y_pred):
    return np.mean(y_true == np.sign(y_pred))


class SupportVectorMachine:
    def __init__(self, C=1, epochs=500, batch_size=125):
        self.C = C
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.model = support_vector_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # @tf.function
    def loss(self, y_true, y_pred):
        return self.C*tf.keras.losses.hinge(y_true, y_pred)

    @tf.function
    def train_step(self, X_batch, Y_batch):
        with tf.GradientTape(persistent=True) as tape:
            pred = self.model(X_batch)
            loss = self.loss(Y_batch, pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        return loss

    def fit(self, train, test):
        X, y = train
        X_test, y_test = test
        self.num_batches = len(X)//self.batch_size
        for epoch in range(self.num_epochs):
            for ind in range(self.num_batches):
                X_batch = X[(ind*self.batch_size):(ind+1)*self.batch_size]
                y_batch = y[(ind*self.batch_size):(ind+1)*self.batch_size]
                loss = self.train_step(X_batch, y_batch).numpy()

            if epoch % 10 == 0:
                X_train_batch = X[(ind*self.batch_size):(ind+1)*self.batch_size]
                y_train_batch = y[(ind*self.batch_size):(ind+1)*self.batch_size]
                pred_train_batch = self.model(X_train_batch)

                X_test_batch = X_test[int((ind*self.batch_size)/TRAIN_RATIO*(1-TRAIN_RATIO)):int(
                    (ind+1)*self.batch_size/TRAIN_RATIO*(1-TRAIN_RATIO))]
                y_test_batch = y_test[int((ind*self.batch_size)/TRAIN_RATIO*(1-TRAIN_RATIO)):int(
                    (ind+1)*self.batch_size/TRAIN_RATIO*(1-TRAIN_RATIO))]
                pred_test_batch = self.model(X_test_batch)
                print("[", epoch+1, "/", self.num_epochs, "]", "loss:", round(loss, 4), 'Train Accuracy:', accuracy(
                    y_train_batch, pred_train_batch), 'Test Accuract:', accuracy(y_test_batch, pred_test_batch))
        return self.model


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]


if __name__ == '__main__':
    obj = Dataset()
    train, test = obj.get_data()

    reg = SupportVectorMachine()
    model = reg.fit(train, test)

    X, y = train
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    fig, ax = plt.subplots()

    x0_1, x0_2 = np.min(X0), np.max(X0)
    w, b = model.get_weights()
    x1_1 = get_hyperplane_value(x0_1, w, b, 0)
    x1_2 = get_hyperplane_value(x0_2, w, b, 0)

    x1_1_m = get_hyperplane_value(x0_1, w, b, -1)
    x1_2_m = get_hyperplane_value(x0_2, w, b, -1)

    x1_1_p = get_hyperplane_value(x0_1, w, b, 1)
    x1_2_p = get_hyperplane_value(x0_2, w, b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('x_1')
    ax.set_xlabel('x_0')
    plt.show()
