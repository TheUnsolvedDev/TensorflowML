import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU number')
args = parser.parse_args()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[args.gpu], 'GPU')


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='conv_autoen.h5', save_weights_only=True, monitor='loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]

LATENT_DIM = 16


def process_images(image, label):
    image = tf.expand_dims(image, axis=-1)/255
    label = tf.expand_dims(label, axis=-1)/255
    return image, label


def encoder(input_shape=(28, 28, 1), latent_dim=LATENT_DIM):
    encoder_inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
        32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = tf.keras.layers.Conv2D(
        64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(LATENT_DIM, activation="relu")(x)
    return tf.keras.Model(encoder_inputs, x, name="encoder")


def decoder(input_shape=(LATENT_DIM,)):
    latent_inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(
        64, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(
        32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(
        1, 3, activation="sigmoid", padding="same")(x)
    return tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")


def ConvAutoEncoder():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = encoder()(inputs)
    outputs = decoder()(x)
    return tf.keras.Model(inputs, outputs)


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


if __name__ == "__main__":
    model = ConvAutoEncoder()
    model.summary(expand_nested=True)

    tf.keras.utils.plot_model(
        model, to_file=ConvAutoEncoder.__name__+'.png', show_shapes=True, expand_nested=True)
    model.compile(optimizer='adam', loss='mse')

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

    model.fit(train_ds, epochs=10, shuffle=True,
              validation_data=validation_ds, callbacks=callbacks)
    test(model, test_ds)
