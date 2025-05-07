import tensorflow as tf
import argparse

from model import *
from dataset import *
from config import *



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
    model_fn = GAN
    parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU number')
    parser.add_argument('--type', type=str, default='cifar10',
                        help='Dataset type', choices=['cifar10', 'fashion_mnist', 'mnist', 'cifar100'])
    args = parser.parse_args()

    # GPU Setup
    setup_gpu(args.gpu)
    dataset = Dataset()
    train_ds, test_ds, channels = dataset.load_data(args.type)
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())  # Call function from model.py
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    print(
        f'Training on dataset {args.type} with {strategy.num_replicas_in_sync} devices')

    model = GAN(strategy=strategy, input_shape=(
        IMAGE_SIZE[0], IMAGE_SIZE[1], channels), latent_dim=(LATENT_DIM,), batch_size=BATCH_SIZE)
    model.fit(train_ds, epochs=EPOCHS, path=args.type,callbacks=None)


if __name__ == '__main__':
    main()
