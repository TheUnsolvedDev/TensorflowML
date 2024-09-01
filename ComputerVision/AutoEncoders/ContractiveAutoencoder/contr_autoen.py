import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU number')
args = parser.parse_args()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[args.gpu], 'GPU')
NOISE = 0.25

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='vanilla_autoen.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def encoder(shape=(784,)):
    inputs = tf.keras.layers.Input(shape)
    flatten = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(
        128, activation='relu')(flatten)
    return tf.keras.Model(inputs, outputs)


def decoder(shape=(128,)):
    inputs = tf.keras.layers.Input(shape)
    outputs = tf.keras.layers.Dense(784, activation='sigmoid')(inputs)
    return tf.keras.Model(inputs, outputs)


def process_images(image, label):
    image = tf.reshape(image, (-1, ))/255
    label = tf.reshape(label, (-1, ))/255
    return image, label


def save_images(images, path='Plots'):
    # plt.figure(figsize=(10, 10))
    imgs = np.hstack([images[0].numpy().reshape(28, 28),
                     images[1].numpy().reshape(28, 28)])
    plt.imshow(imgs)
    plt.savefig(path)


def test(model, test_ds):
    images, labels = next(iter(test_ds))
    for ind, (image, label) in enumerate(zip(images[:32], labels[:32])):
        image = tf.expand_dims(image, axis=0)
        pred = model(image)
        save_images([pred, label], path='Plots/predictions'+str(ind)+'.png')


class ContractiveAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(ContractiveAutoEncoder, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        self.loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.Lambda = 0.1

    def call(self, inputs, training=False):
        aux_x = self.encoder(inputs)
        x = self.decoder(aux_x)
        return x

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape1:
            pred = self.call(x, training=True)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                latent = self.encoder(x)
            grads_images = tf.reduce_mean(
                tf.reduce_mean(tf.square(tape2.gradient(latent, x))))
            loss = tf.keras.losses.mean_squared_error(
                y, pred) + grads_images*self.Lambda

        grads = tape1.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"train_loss": self.loss_tracker.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        pred = self.call(x, training=False)
        loss = tf.keras.losses.mean_squared_error(
            y, pred)
        self.val_loss_tracker.update_state(loss)
        return {'loss': self.val_loss_tracker.result()}


if __name__ == '__main__':
    model = ContractiveAutoEncoder()
    model.build((None, 784))
    model.summary(expand_nested=True)
    tf.keras.utils.plot_model(
        model, to_file=ContractiveAutoEncoder.__name__+'.png', show_shapes=True, expand_nested=True)
    model.compile(optimizer='adam')

    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.mnist.load_data()

    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_images))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_images))
    validation_ds = tf.data.Dataset.from_tensor_slices(
        (validation_images, validation_images))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(
        validation_ds).numpy()
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)

    train_ds = (train_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=128, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=128, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=128, drop_remainder=True))

    model.fit(train_ds, epochs=10,
              validation_data=validation_ds, callbacks=callbacks)
    test(model, test_ds)
