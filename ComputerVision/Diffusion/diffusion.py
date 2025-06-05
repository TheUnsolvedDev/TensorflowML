import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ============ Configs ============
TIMESTEPS = 1000
IMG_SHAPE = (28, 28, 1)
SAVE_INTERVAL = 1  # Save generated images every N epochs
OUTPUT_DIR = "generated_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ Diffusion Schedule ============

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

class Diffusion:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps)
        self.betas = tf.convert_to_tensor(betas, dtype=tf.float32)
        self.alphas = 1. - self.betas
        self.alpha_hat = tf.math.cumprod(self.alphas, axis=0)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x0))
        sqrt_alpha_hat = tf.gather(tf.sqrt(self.alpha_hat), t)
        sqrt_one_minus_alpha_hat = tf.gather(tf.sqrt(1. - self.alpha_hat), t)
        sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, [-1, 1, 1, 1])
        sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, [-1, 1, 1, 1])
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise

    def sample(self, model, shape):
        x = tf.random.normal(shape)
        for t in reversed(range(self.timesteps)):
            t_batch = tf.fill([x.shape[0]], t)
            pred_noise = model(x, training=False)

            beta = self.betas[t]
            alpha = self.alphas[t]
            alpha_hat = self.alpha_hat[t]

            # Make sure all are float32
            beta = tf.cast(beta, tf.float32)
            alpha = tf.cast(alpha, tf.float32)
            alpha_hat = tf.cast(alpha_hat, tf.float32)

            x = (1. / tf.sqrt(alpha)) * (x - beta / tf.sqrt(1. - alpha_hat) * pred_noise)
            if t > 0:
                noise = tf.random.normal(shape)
                x += tf.sqrt(beta) * noise
        return tf.clip_by_value(x, -1.0, 1.0)


# ============ UNet-like Model ============

def build_unet(img_shape):
    inputs = tf.keras.Input(shape=img_shape)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
    return tf.keras.Model(inputs, x)

# ============ Data Loader ============

def prepare_mnist():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) / 127.5) - 1.0
    x_train = np.expand_dims(x_train, -1)
    return tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(128)

# ============ Sampling Visualization ============

def save_images(images, epoch, output_dir):
    images = (images + 1.0) / 2.0  # Rescale to [0, 1]
    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i, ..., 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_epoch_{epoch}.png"))
    plt.close()

# ============ Training Loop ============

def train(model, diffusion, dataset, epochs=10, save_interval=1):
    optimizer = tf.keras.optimizers.Adam(1e-4)
    mse = tf.keras.losses.MeanSquaredError()

    for epoch in range(1, epochs + 1):
        for step, x in enumerate(dataset):
            t = tf.random.uniform((x.shape[0],), minval=0, maxval=diffusion.timesteps, dtype=tf.int32)
            noise = tf.random.normal(shape=tf.shape(x))
            x_noisy, noise = diffusion.add_noise(x, t, noise)

            with tf.GradientTape() as tape:
                pred_noise = model(x_noisy, training=True)
                loss = mse(noise, pred_noise)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.numpy():.4f}")

        if epoch % save_interval == 0:
            samples = diffusion.sample(model, shape=(16, *IMG_SHAPE))
            save_images(samples, epoch, OUTPUT_DIR)

# ============ Main ============
if __name__ == "__main__":
    dataset = prepare_mnist()
    model = build_unet(IMG_SHAPE)
    diffusion = Diffusion(timesteps=TIMESTEPS)
    train(model, diffusion, dataset, epochs=10, save_interval=SAVE_INTERVAL)
