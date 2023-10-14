import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import sys
from silence_tensorflow import silence_tensorflow

silence_tensorflow()


parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU number')
args = parser.parse_args()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[args.gpu], 'GPU')

ITERATION = 10_000
Z_DIM = 100
BATCH_SIZE = 256
BUFFER_SIZE = 6000
D_LR = 0.0002
G_LR = 0.002
IMAGE_SHAPE = [28, 28, 1]
RANDOM_SEED = 42
GP_WEIGHT = 10.0
EVERY_STEP = 50
N_CRITIC = 5

callbacks = [
    # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_deep_cnn_gan.h5', save_weights_only=True, monitor='loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=50, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


class Dataset:
    def __init__(self,):
        (self.train_data, train_labels), (self.test_data,
                                          test_labels) = tf.keras.datasets.mnist.load_data()
        self.train_data = self.train_data.reshape(-1, 28, 28, 1)/255.0
        self.test_data = self.test_data.reshape(-1, 28, 28, 1)/255.0

    def get_data(self):
        self.train_data = tf.data.Dataset.from_tensor_slices(
            self.train_data).prefetch(tf.data.AUTOTUNE).shuffle(BUFFER_SIZE)
        self.test_data = tf.data.Dataset.from_tensor_slices(
            self.test_data).prefetch(tf.data.AUTOTUNE).shuffle(BUFFER_SIZE)
        return self.train_data.batch(BATCH_SIZE), self.test_data.batch(BATCH_SIZE)


def make_discriminator():
    inputs = tf.keras.layers.Input(shape=(IMAGE_SHAPE))
    x = tf.keras.layers.Conv2D(
        64, (5, 5), strides=(2, 2), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(
        128, (5, 5), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    outputs = tf.keras.layers.Dense(1, activation=None)(x)
    return tf.keras.models.Model(inputs, outputs)


def make_generator():
    inputs = tf.keras.layers.Input(shape=(Z_DIM,))

    x = tf.keras.layers.Dense(256*7*7)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Reshape((7, 7, 256))(x)

    x = tf.keras.layers.Conv2DTranspose(
        128, (5, 5), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    outputs = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(
        2, 2), padding='same', use_bias=False, activation='sigmoid')(x)
    return tf.keras.models.Model(inputs, outputs)


criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def d_loss_fn(real_logits, fake_logits):
    real_loss = criterion(tf.ones_like(real_logits), real_logits)
    fake_loss = criterion(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss


def g_loss_fn(fake_logits):
    return criterion(tf.ones_like(fake_logits), fake_logits)


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=16, latent_dim=Z_DIM):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        if epoch % (ITERATION//20) != 0:
            return
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim))
        generated_images = self.model.gen(random_latent_vectors).numpy()
        generated_images *= 255.0

        images = [np.hstack(generated_images[i:i+4])
                  for i in range(0, self.num_img, 4)]
        images = np.vstack(images)

        img = tf.keras.utils.array_to_img(images)
        img.save("plots/generated_img_{epoch}.png".format(epoch=epoch))

    def on_train_end(self, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim))
        generated_images = self.model.gen(random_latent_vectors).numpy()
        generated_images *= 255.0

        images = [np.hstack(generated_images[i:i+4])
                  for i in range(0, self.num_img, 4)]
        images = np.vstack(images)

        img = tf.keras.utils.array_to_img(images)
        img.save("plots/generated_img_final.png")


class DeepCNNGAN(tf.keras.Model):
    def __init__(self,):
        super(DeepCNNGAN, self).__init__()
        self.gen = make_generator()
        self.disc = make_discriminator()
        os.makedirs('plots', exist_ok=True)
        os.makedirs('first_gan_weights', exist_ok=True)

    def compile(self, g_loss, d_loss, g_opt, d_opt):
        super().compile()
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_opt = g_opt
        self.d_opt = d_opt

    def call(self, x):
        self.batch_size = tf.shape(x)[0]

    def summary(self):
        self.gen.summary()
        self.disc.summary()

    # @tf.function
    def train_step(self, real_images):
        self(real_images)
        noise = tf.random.normal(
            [self.batch_size, Z_DIM])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.gen(noise)
            disc_fake_pred = self.disc(generated_images)
            disc_real_pred = self.disc(real_images)

            disc_loss = self.d_loss(disc_real_pred, disc_fake_pred)
            gen_loss = self.g_loss(disc_fake_pred)
        disc_gradients = disc_tape.gradient(
            disc_loss, self.disc.trainable_variables)
        self.d_opt.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables))

        gen_gradients = gen_tape.gradient(
            gen_loss, self.gen.trainable_variables)
        self.g_opt.apply_gradients(
            zip(gen_gradients, self.gen.trainable_variables))

        return {'loss': 0.5*(disc_loss+gen_loss), 'G Loss:': gen_loss, 'D Loss:': disc_loss}


if __name__ == '__main__':
    data = Dataset()
    train, test = data.get_data()

    model = DeepCNNGAN()
    model.summary()
    gen_optimizer = tf.keras.optimizers.SGD(G_LR)
    disc_optimizer = tf.keras.optimizers.SGD(D_LR)
    model.compile(g_loss=g_loss_fn, d_loss=d_loss_fn,
                  g_opt=gen_optimizer, d_opt=disc_optimizer)
    model.fit(train, epochs=ITERATION, callbacks=callbacks+[GANMonitor()])
