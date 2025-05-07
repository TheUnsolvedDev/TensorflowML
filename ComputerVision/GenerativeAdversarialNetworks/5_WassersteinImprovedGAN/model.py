import tensorflow as tf
import matplotlib.pyplot as plt
import os,sys
import tqdm
from config import *

# Create images folder
os.makedirs("images", exist_ok=True)


class GAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, latent_dim, batch_size, gp_weight=10.0):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.global_batch_size = batch_size * self.strategy.num_replicas_in_sync
        self.gp_weight = gp_weight

        with self.strategy.scope():
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()

            self.generator_optimizer = tf.keras.optimizers.Adam(
                GENERATOR_LEARNING_RATE)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                DISCRIMINATOR_LEARNING_RATE)

        self.generator.build(input_shape=(None, latent_dim[0]))
        self.discriminator.build(input_shape=(None, *input_shape))

        self.generator.summary()
        self.discriminator.summary()

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=(self.latent_dim[0],))

        # Start from a small feature map size depending on input resolution
        init_height = self.input_shape[0] // 4
        init_width = self.input_shape[1] // 4
        init_channels = 256

        x = tf.keras.layers.Dense(
            init_height * init_width * init_channels, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Reshape(
            (init_height, init_width, init_channels))(x)

        x = tf.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        # Final layer outputs the desired channels (1 or 3) and final size
        x = tf.keras.layers.Conv2DTranspose(self.input_shape[2], (5, 5), strides=(
            1, 1), padding='same', use_bias=False, activation='tanh')(x)

        outputs = x
        return tf.keras.Model(inputs, outputs, name="generator")

    def build_discriminator(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)

        outputs = x
        return tf.keras.Model(inputs, outputs, name="discriminator")

    def gradient_penalty(self, real_images, generated_images):
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        interpolated = alpha * real_images + (1 - alpha) * generated_images
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = gp_tape.gradient(pred, interpolated)
        grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grads_norm - 1.0))
        return gradient_penalty

    @tf.function
    def train_discriminator_step(self, real_images):
        noise = tf.random.normal([tf.shape(real_images)[0], self.latent_dim[0]])
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_cost = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gp = self.gradient_penalty(real_images, generated_images)
            disc_loss = disc_cost + self.gp_weight * gp
        gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables))
        return disc_loss

    @tf.function
    def train_generator_step(self):
        noise = tf.random.normal([self.global_batch_size, self.latent_dim[0]])
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = -tf.reduce_mean(fake_output)
        gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables))
        return gen_loss
    
    @tf.function
    def distributed_gen_train_step(self):
        per_replica_gen_loss = self.strategy.run(self.train_generator_step, args=())
        gen_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_gen_loss, axis=None)
        return gen_loss

    @tf.function
    def distributed_disc_train_step(self, images):
        per_replica_disc_loss = self.strategy.run(self.train_discriminator_step, args=(images,))
        disc_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_disc_loss, axis=None)
        return disc_loss

    def generate_and_save_images(self, epoch, num_examples=16, path='folder'):
        path = f'images/{path}'
        os.makedirs(path, exist_ok=True)
        noise = tf.random.normal([num_examples, self.latent_dim[0]])
        generated_images = self.generator(noise, training=False)
        generated_images = (generated_images + 1) / 2.0  # Rescale [0,1]

        fig, axs = plt.subplots(4, 4, figsize=(4, 4))
        idx = 0
        for i in range(4):
            for j in range(4):
                img = generated_images[idx]
                if self.input_shape[2] == 1:
                    img = img[:, :, 0]
                    axs[i, j].imshow(img, cmap='gray')
                else:
                    axs[i, j].imshow(img)
                axs[i, j].axis('off')
                idx += 1

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(f"{path}/image_at_epoch_{epoch:03d}.png")
        plt.close()

    def fit(self, dataset, epochs, path='folder', callbacks=None):
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            callback.set_model(self)
            callback.on_train_begin()

        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            disc_loss = tf.constant(0.0)
            for step, image_batch in enumerate(dataset):
                gen_loss = self.distributed_gen_train_step()
                if step % 5 == 0:
                    disc_loss = self.distributed_disc_train_step(image_batch)
                print(
                    f'\rEpoch [{step}/{epoch+1}], Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}',end='')
                sys.stdout.flush()
            print()
            self.generate_and_save_images(epoch+1, path=path)

            logs = {"gen_loss": gen_loss.numpy(), "disc_loss": disc_loss.numpy()}
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end()


if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())
    gan = GAN(strategy=strategy, input_shape=(
        IMAGE_SIZE[0], IMAGE_SIZE[1], 1), latent_dim=(LATENT_DIM,), batch_size=BATCH_SIZE)
