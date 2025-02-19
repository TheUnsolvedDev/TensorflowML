import silence_tensorflow.auto
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import os

from config import *
from dataset import *
from model import *


class SaveImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, seed=42, save_dir="saved_images", n_epochs=1, dataset_type="default"):
        super(SaveImageCallback, self).__init__()
        self.seed = seed
        self.n_epochs = n_epochs
        self.latent_dim = LATENT_DIM
        self.save_dir = os.path.join(save_dir, dataset_type)
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n_epochs == 0:
            tf.random.set_seed(self.seed)
            noise = tf.random.normal(shape=(1, self.latent_dim))
            prediction = self.model.generator(noise, training=False)
            prediction = np.array(prediction, dtype=np.uint8)
            prediction = np.squeeze(prediction)

            plt.imshow(prediction, cmap='gray')  # Adjust cmap as needed
            plt.axis('off')
            save_path = os.path.join(self.save_dir, f"epoch_{epoch+1}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


def setup_gpu(gpu_id):
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    if gpu_id == -1:
        tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
        print("Using all available GPUs")
    elif 0 <= gpu_id < len(physical_devices):
        tf.config.experimental.set_visible_devices(
            physical_devices[gpu_id], 'GPU')
        print(f"Using GPU: {gpu_id}")
    else:
        print("Invalid GPU ID. Defaulting to CPU.")


def main():
    model_fn = NNGAN
    parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--type', type=str, default='cifar10',
                        help='Dataset type', choices=['cifar10', 'fashion_mnist', 'mnist', 'cifar100'])
    args = parser.parse_args()

    # GPU Setup
    setup_gpu(args.gpu)

    dataset = Dataset()
    train_ds, test_ds, channels = dataset.load_data(args.type)

    # Load strategy from `model.py`
    strategy = tf.distribute.MirroredStrategy()  # Call function from model.py
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    print(
        f'Training on dataset {args.type} with {strategy.num_replicas_in_sync} devices')
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{model_fn.__name__}_' + args.type + '.weights.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs_{args.type}_{model_fn.__name__}', histogram_freq=1, write_graph=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
        SaveImageCallback(n_epochs=1, dataset_type=args.type)
    ]

    with strategy.scope():
        model = NNGAN(strategy, gan_input_shape=(LATENT_DIM,), disc_input_shape=(
            IMAGE_SIZE[0], IMAGE_SIZE[1], channels))
        model.build(input_shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], channels))
        model.summary(expand_nested=True)
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=GENERATOR_LEARNING_RATE)
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=DISCRIMINATOR_LEARNING_RATE)
        model.compile(gen_loss=generator_loss, disc_loss=discriminator_loss,gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)
        model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)


if __name__ == '__main__':
    main()
