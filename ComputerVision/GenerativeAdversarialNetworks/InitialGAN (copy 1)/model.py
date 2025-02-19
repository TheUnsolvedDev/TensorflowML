import tensorflow as tf
import numpy as np
from config import *


def generator_loss(fake):
    loss_fn = tf.keras.losses.binary_crossentropy
    return loss_fn(tf.ones_like(fake), fake)


def discriminator_loss(real, fake):
    loss_fn = tf.keras.losses.binary_crossentropy
    real_loss = loss_fn(tf.ones_like(real), real)
    fake_loss = loss_fn(tf.zeros_like(fake), fake)
    return 0.5 * (real_loss + fake_loss)


def generator_model(input_shape=(64,), output_shape=(32, 32, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, activation='leaky_relu')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(np.prod(output_shape), activation='sigmoid')(x)
    x = tf.keras.layers.Reshape(output_shape)(x)
    outputs = tf.keras.layers.Lambda(lambda x: x * 255.0)(x)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name='generator')



def discriminator_model(input_shape=(32, 32, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='leaky_relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name='discriminator')


class NNGAN(tf.keras.Model):
    def __init__(self, strategy, gan_input_shape, disc_input_shape):
        super(NNGAN, self).__init__()
        self.strategy = strategy
        self.gan_input_shape = gan_input_shape
        self.disc_input_shape = disc_input_shape
        # with strategy.scope():
        self.generator = generator_model(
            input_shape=gan_input_shape, output_shape=disc_input_shape)
        self.discriminator = discriminator_model(input_shape=disc_input_shape)

        self.alpha_gen = GENERATOR_LEARNING_RATE
        self.alpha_disc = DISCRIMINATOR_LEARNING_RATE

    def call(self, inputs, training=False):
        noise = tf.random.normal(
            shape=(tf.shape(inputs)[0], self.gan_input_shape[0]))
        return self.generator(noise, training=training)

    def compile(self, gen_loss, disc_loss, gen_optimizer, disc_optimizer):
        super(NNGAN, self).compile()
        # with self.strategy.scope():
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.generator_optimizer = gen_optimizer
        self.discriminator_optimizer = disc_optimizer

    # @tf.function
    def distributed_train_step(self, real_images):
        per_replica_losses = self.strategy.experimental_run(
            self.train_step, args=(real_images,))
        replica_gen_loss = per_replica_losses['G Loss']
        replica_disc_loss = per_replica_losses['D Loss']
        total_g_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, replica_gen_loss, axis=None)
        total_d_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, replica_disc_loss, axis=None)
        total_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses['loss'], axis=None)
        return {'loss': total_loss, 'D Loss': total_d_loss, 'G Loss': total_g_loss}

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.gan_input_shape[0]))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            tf.print(tf.shape(real_output), tf.shape(fake_output))

            gen_loss_value = self.gen_loss(fake_output)
            disc_loss_value = self.disc_loss(real_output, fake_output)
            tf.print(gen_loss_value, disc_loss_value)

        gradients_of_generator = gen_tape.gradient(
            gen_loss_value, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss_value, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss_value = self.gen_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss_value, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        return {'loss': 0.5 * (gen_loss_value + disc_loss_value), 'D Loss': disc_loss_value, 'G Loss': gen_loss_value}


if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy()
    model = NNGAN(strategy)
    model.compile(gen_loss=generator_loss, disc_loss=discriminator_loss,
                  gen_optimizer=tf.keras.optimizers.Adam(1e-4), disc_optimizer=tf.keras.optimizers.Adam(1e-4))
    model.build(input_shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.summary(expand_nested=True)

    tf.keras.utils.plot_model(
        model.generator, expand_nested=True, show_shapes=True, to_file='generator.png')
    tf.keras.utils.plot_model(model.discriminator, expand_nested=True,
                              show_shapes=True, to_file='discriminator.png')

    # tf.keras.utils.plot_model(model,expand_nested=True, show_shapes=True)
