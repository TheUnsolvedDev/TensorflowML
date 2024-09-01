import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import silence_tensorflow.auto

from params import *
from model import *
from dataset import *


parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU number')
parser.add_argument('--name', type=str, default='vangogh2photo',
                    help='Name of the Dataset to select')
args = parser.parse_args()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if args.gpu >= 0 and args.gpu <= 3:
    tf.config.experimental.set_visible_devices(
        physical_devices[args.gpu], 'GPU')
else:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, enable=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='Train_loss', patience=25),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='cycle_gan'+args.name+'.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.LearningRateScheduler(
        schedule=lambda epoch: 0.005 * (0.995 ** (epoch//10)))
]


def update_images(test_data):
    class Updates(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 25 == 0:
                for data in test_data.take(1):
                    pred_G, pred_F = cycle_gan_model(data)
                    real_imgs1 = np.hstack(data[0][:4]*255).astype(np.uint8)
                    pred_imgs_G = np.hstack(pred_G[:4]*255).astype(np.uint8)
                    real_imgs2 = np.hstack(data[1][:4]*255).astype(np.uint8)
                    pred_imgs_F = np.hstack(pred_F[:4]*255).astype(np.uint8)
                    pred = np.vstack(
                        [real_imgs1, pred_imgs_G, real_imgs2, pred_imgs_F])
                    plt.imshow(pred)
                    plt.savefig('Plots/'+args.name+'/Updates{}.png'.format(epoch))
            return super().on_epoch_end(epoch, logs)
    return Updates()


def save_imgs(epoch, generator, real_x):
    gene_imgs = generator(real_x, training=False)

    gene_imgs = ((gene_imgs.numpy() + 1) * 127.5).astype(np.uint8)
    real_x = ((real_x.numpy() + 1) * 127.5).astype(np.uint8)

    fig = plt.figure(figsize=(8, 12))
    tmp = 0
    for i in range(0, 4):
        plt.subplot(4, 2, i + 1 + tmp)
        plt.imshow(real_x[i])
        plt.axis('off')
        plt.subplot(4, 2, i + 2 + tmp)
        plt.imshow(gene_imgs[i])
        plt.axis('off')
        tmp += 1

    fig.savefig("trial_images/result_{}.png".format(str(epoch).zfill(5)))


def generator_loss_fn(fake):
    fake_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(fake), fake)
    return tf.reduce_mean(fake_loss)


def discriminator_loss_fn(real, fake):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real), real)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake), fake)
    return tf.reduce_mean((real_loss + fake_loss) * 0.5)


class CycleGan(tf.keras.Model):
    def __init__(self):
        super(CycleGan, self).__init__()
        self.gen_G = get_resnet_generator(name="generator_G")
        self.gen_F = get_resnet_generator(name="generator_F")
        self.disc_X = get_discriminator(name="discriminator_X")
        self.disc_Y = get_discriminator(name="discriminator_Y")
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.5

        self.gen_G_optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=2e-4, beta_1=0.5)
        self.gen_F_optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=2e-4, beta_1=0.5)
        self.disc_X_optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=2e-4, beta_1=0.5)
        self.disc_Y_optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=2e-4, beta_1=0.5)
        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn

    def call(self, inputs):
        real_x, real_y = inputs
        return self.gen_G(real_x), self.gen_F(real_y)

    # @tf.function
    def train_step(self, inputs):
        real_x, real_y = inputs
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.gen_G(real_x, training=True)
            fake_x = self.gen_F(real_y, training=True)

            cycled_x = self.gen_F(fake_y, training=True)
            cycled_y = self.gen_G(fake_x, training=True)

            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            cycle_loss_G = tf.reduce_mean(tf.keras.losses.mean_absolute_error(
                real_y, cycled_y) * self.lambda_cycle)
            cycle_loss_F = tf.reduce_mean(tf.keras.losses.mean_absolute_error(
                real_x, cycled_x) * self.lambda_cycle)

            id_loss_G = tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)
        disc_X_grads = tape.gradient(
            disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(
            disc_Y_loss, self.disc_Y.trainable_variables)

        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )
        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }


if __name__ == '__main__':
    data, test_data = dataset(BATCH_SIZE,dataset_name = args.name)
    # Get the generators
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope() as scope:
        cycle_gan_model = CycleGan()
        cycle_gan_model.compile(loss=None)

        cycle_gan_model.fit(
            data, epochs=int(1e+5), callbacks=callbacks+[update_images(test_data)])

    # cycle_gan_model = CycleGan()
    # for data in test_data.take(1):
    #     pred_G, pred_F = cycle_gan_model(data)
    #     real_imgs1 = np.hstack(data[0][:4]*255).astype(np.uint8)
    #     pred_imgs_G = np.hstack(pred_G[:4]*255).astype(np.uint8)
    #     real_imgs2 = np.hstack(data[1][:4]*255).astype(np.uint8)
    #     pred_imgs_F = np.hstack(pred_G[:4]*255).astype(np.uint8)
    #     pred = np.vstack([real_imgs1, pred_imgs_G, real_imgs2, pred_imgs_F])
    #     print(pred.shape)

    # # imgs = np.hstack([a[0]*255,b[0]*255]).astype(np.uint8)
    # plt.imshow(pred)
    # plt.savefig('plot.png')
    # test_model(test_data)
