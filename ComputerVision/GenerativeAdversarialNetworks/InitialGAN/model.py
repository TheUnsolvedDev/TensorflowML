import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *


def generator_model(input_shape=(LATENT_DIM,), output_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, use_bias=False)(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(np.prod(output_shape),
                              use_bias=False, activation='sigmoid')(x)
    x = tf.keras.layers.Reshape(output_shape)(x)
    outputs = tf.keras.layers.Lambda(lambda x: x*255.0)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='generator')
    return model


def discriminator_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Lambda(lambda x: x/255.0)(x)

    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(64, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(32, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='discriminator')
    return model


class DeepNNGAN:
    def __init__(self, strategy, batch_size=BATCH_SIZE, gen_input_shape=(LATENT_DIM,), disc_input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)):
        self.strategy = strategy
        self.batch_size = batch_size
        self.gen_input_shape = gen_input_shape
        self.disc_input_shape = disc_input_shape
        self.batch_per_replica = int(
            batch_size / strategy.num_replicas_in_sync)

        self.generator = generator_model(
            input_shape=self.gen_input_shape, output_shape=self.disc_input_shape)
        self.discriminator = discriminator_model(self.disc_input_shape)
        self.generator_optimizer = tf.keras.optimizers.Adam(
            GENERATOR_LEARNING_RATE)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            DISCRIMINATOR_LEARNING_RATE)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)

    def generator_loss(self, fake_output):
        loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        return loss/self.batch_size

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        total_loss = (real_loss + fake_loss)/self.batch_size
        return total_loss

    # @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.gen_input_shape[0]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(
                generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        return {"d_loss": disc_loss, "g_loss": gen_loss}

    @tf.function
    def distributed_train_step(self, dataset):
        per_replica_losses = self.strategy.run(
            self.train_step, args=(dataset,))
        replica_gen_loss = per_replica_losses['g_loss']
        replica_disc_loss = per_replica_losses['d_loss']

        total_g_loss = self.strategy.reduce(
            "sum", replica_gen_loss, axis=0)
        total_d_loss = self.strategy.reduce(
            "sum", replica_disc_loss, axis=0)
        return {"d_loss": total_d_loss, "g_loss": total_g_loss}

    def fit(self, train_dataset, epochs=100, callbacks=None):
        for _ in range(epochs):
            for images in train_dataset:
                loss = self.distributed_train_step(images)
                print(
                    f'Epoch: {_+1}, Discriminator Loss: {loss["d_loss"]:.4f}, Generator Loss: {loss["g_loss"]:.4f}')
        print('Training complete')
        



if __name__ == '__main__':
    generator = generator_model()
    discriminator = discriminator_model()
    generator.summary()
    discriminator.summary()
    print('Done')
