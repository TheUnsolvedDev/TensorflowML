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

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='sparse_autoen.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

LATENT_DIM = 128

def encoder(shape=(784,)):
    inputs = tf.keras.layers.Input(shape)
    x = tf.keras.layers.Dense(LATENT_DIM, activation='relu')(inputs)
    outputs = tf.keras.layers.ActivityRegularization(l1=1e-1)(x)
    return tf.keras.Model(inputs, outputs)


def decoder(shape=(LATENT_DIM,)):
    inputs = tf.keras.layers.Input(shape)
    outputs = tf.keras.layers.Dense(784, activation='sigmoid')(inputs)
    return tf.keras.Model(inputs, outputs)


def autoencoder(shape=(784,)):
    inputs = tf.keras.layers.Input(shape)
    encoded = encoder()(inputs)
    decoded = decoder()(encoded)
    return tf.keras.Model(inputs, decoded)


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


if __name__ == '__main__':
    model = autoencoder()
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file=autoencoder.__name__+'.png', show_shapes=True,expand_nested=True)
    model.compile(optimizer='adam', loss='binary_crossentropy')

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

    model.fit(train_ds, epochs=10, shuffle=True, validation_data=validation_ds,callbacks=callbacks)
    test(model, test_ds)
