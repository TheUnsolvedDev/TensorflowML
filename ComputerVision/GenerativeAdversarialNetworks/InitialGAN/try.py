import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the Generator as a functional model
def build_generator(latent_dim):
    inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(28 * 28 * 1, activation='tanh')(x)
    outputs = layers.Reshape((28, 28, 1))(outputs)
    return keras.Model(inputs, outputs, name="Generator")

# Define the Discriminator as a functional model
def build_discriminator():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Flatten()(inputs)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs, name="Discriminator")

# Define the GAN model as a subclassed keras.Model
class GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, gen_optimizer, disc_optimizer, loss_fn):
        super(GAN, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.loss_fn = loss_fn

    def call(self, inputs, training=False):
        if not isinstance(inputs, tf.Tensor):
            raise TypeError("GAN input must be a Tensor.")
        if inputs.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected input shape (*, {self.latent_dim}), but got {inputs.shape}")
        return self.generator(inputs, training=training)

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # Generate fake images
        fake_images = self.generator(random_latent_vectors, training=True)
        
        # Train discriminator
        with tf.GradientTape() as tape:
            real_preds = self.discriminator(real_images, training=True)
            fake_preds = self.discriminator(fake_images, training=True)
            real_loss = self.loss_fn(tf.ones_like(real_preds), real_preds)
            fake_loss = self.loss_fn(tf.zeros_like(fake_preds), fake_preds)
            disc_loss = (real_loss + fake_loss) / 2
        grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            fake_preds = self.discriminator(generated_images, training=True)
            gen_loss = self.loss_fn(tf.ones_like(fake_preds), fake_preds)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        
        return {"disc_loss": disc_loss, "gen_loss": gen_loss}

# Enable multi-GPU training
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    gan = GAN(generator, discriminator, latent_dim)
    gan.compile(
        gen_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        disc_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=keras.losses.BinaryCrossentropy()
    )

# Prepare dataset
def preprocess(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - 0.5) * 2  # Normalize to [-1, 1]
    return image

dataset = tf.keras.datasets.mnist.load_data()
(train_images, _), (_, _) = dataset
train_images = train_images[..., None].astype("float32")

batch_size = 128
dataset = (tf.data.Dataset.from_tensor_slices(train_images)
           .map(lambda x: preprocess(x), num_parallel_calls=tf.data.AUTOTUNE)
           .shuffle(10000)
           .batch(batch_size)
           .prefetch(tf.data.AUTOTUNE))

# Ensure model is built before training
gan.generator.build(input_shape=(None, 100))
gan.discriminator.build(input_shape=(None, 28, 28, 1))
gan.build(input_shape=(None, 100))


# Train model with callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint("gan_model.keras", save_best_only=True)
]

gan.fit(dataset, epochs=50, callbacks=callbacks)
