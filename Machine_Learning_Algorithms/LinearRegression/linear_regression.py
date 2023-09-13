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
        train_data, train_label = tf.gather(self.data, indices[:int(len(
            result)*TRAIN_RATIO)]), tf.gather(result, indices[:int(len(result)*TRAIN_RATIO)])
        test_data, test_label = tf.gather(self.data, indices[int(len(
            result)*TRAIN_RATIO):]), tf.gather(result, indices[int(len(result)*TRAIN_RATIO):])
        print(len(train_data),len(test_data))
        return (train_data, train_label), (test_data, test_label)

# Define the linear regression model


class linear_model(tf.Module):
    def __init__(self):
        self.W = tf.Variable(tf.random.normal(
            shape=(NUM_FEATURES, 1), dtype=tf.float32), name='weight')
        self.b = tf.Variable(tf.zeros(1, dtype=tf.float32), name='bias')

    def __call__(self, x):
        return tf.matmul(x, self.W) + self.b


def r_squared(y_true, y_pred):
    total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    unexplained_error = tf.reduce_sum(tf.square(y_true - y_pred))
    r2 = 1 - (unexplained_error / total_error)
    return r2.numpy()


class LinearRegression:
    def __init__(self) -> None:
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.batch_size = 1024
        self.num_epochs = 100
        self.model = linear_model()

    @tf.function
    def loss(self, pred, y):
        mse = tf.square(y - pred)
        result = tf.reduce_mean(mse)
        return result

    @tf.function
    def train_step(self, X_batch, y_batch):
        with tf.GradientTape(persistent=True) as tape:
            pred = self.model(X_batch)
            loss_val = self.loss(pred, y_batch)

        grads = tape.gradient(loss_val, (self.model.W, self.model.b))
        self.optimizer.apply_gradients(
            zip(grads, (self.model.W, self.model.b)))
        return loss_val

    def fit(self, train, test):
        X, y = train
        X_test, y_test = test
        self.num_batches = len(X)//self.batch_size
        for epoch in tqdm.tqdm(range(self.num_epochs+1)):
            for ind in range(self.num_batches):
                X_batch = X[(ind*self.batch_size):(ind+1)*self.batch_size]
                y_batch = y[(ind*self.batch_size):(ind+1)*self.batch_size]
                loss = self.train_step(X_batch, y_batch).numpy()

            if epoch % 10 == 0:
                X_train_batch = X[(ind*self.batch_size):(ind+1)*self.batch_size]
                y_train_batch = y[(ind*self.batch_size):(ind+1)*self.batch_size]
                pred_train_batch = self.model(X_train_batch)
                
                X_test_batch = X_test[int((ind*self.batch_size)/TRAIN_RATIO*(1-TRAIN_RATIO)):int((ind+1)*self.batch_size/TRAIN_RATIO*(1-TRAIN_RATIO))]
                y_test_batch = y_test[int((ind*self.batch_size)/TRAIN_RATIO*(1-TRAIN_RATIO)):int((ind+1)*self.batch_size/TRAIN_RATIO*(1-TRAIN_RATIO))]
                pred_test_batch = self.model(X_test_batch)
                print("loss:", round(loss, 4), 'Train R2_score:', r_squared(
                    y_train_batch, pred_train_batch), 'Test R2_score:', r_squared(y_test_batch, pred_test_batch))
        return self.model


if __name__ == '__main__':
    obj = Dataset()
    train, test = obj.get_data()

    reg = LinearRegression()
    reg.fit(train, test)
