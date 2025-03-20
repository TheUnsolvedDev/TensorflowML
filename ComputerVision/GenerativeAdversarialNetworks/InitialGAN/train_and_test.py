import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from config import *
from model import *
from dataset import *


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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU number')
    parser.add_argument('--type', type=str, default='mnist',
                        help='Dataset type', choices=['cifar10', 'fashion_mnist', 'mnist', 'cifar100'])
    args = parser.parse_args()

    # GPU Setup
    setup_gpu(args.gpu)
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())
    print(
        f'Training on dataset {args.type} with {strategy.num_replicas_in_sync} devices')

    dataset = Dataset(strategy=strategy, batch_size=BATCH_SIZE)
    train_ds, test_ds, channels = dataset.load_data(args.type)
    
    gan = DeepNNGAN(strategy, gen_input_shape=(LATENT_DIM,),
                    disc_input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], channels))
    gan.generator.summary()
    gan.discriminator.summary()

    gan.fit(train_ds, epochs=EPOCHS)
