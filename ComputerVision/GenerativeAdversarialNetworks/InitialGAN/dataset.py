import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from config import *


class Dataset:
    def __init__(self, strategy, batch_size=BATCH_SIZE, img_shape=IMAGE_SIZE):
        self.mnist = tf.keras.datasets.mnist.load_data()
        self.cifar10 = tf.keras.datasets.cifar10.load_data()
        self.fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
        self.cifar100 = tf.keras.datasets.cifar100.load_data()
        self.data_types = ['cifar10', 'fashion_mnist',
                           'mnist',  'cifar100']
        self.strategy = strategy
        self.batch_size = batch_size*strategy.num_replicas_in_sync
        self.img_shape = img_shape

    def process_images(self, image):
        # image = tf.image.per_image_standardization(image)
        image = tf.image.resize(image, self.img_shape)
        image = tf.cast(image, tf.int32)
        return image

    def load_data(self, type='mnist'):
        if type == 'mnist':
            self.channels = 1
            (train_images, train_labels), (test_images, test_labels) = self.mnist
            train_images, test_images = train_images.reshape(
                -1, 28, 28, 1), test_images.reshape(-1, 28, 28, 1)
            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images)).shuffle(10000).batch(self.batch_size, drop_remainder=True).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images)).batch(self.batch_size, drop_remainder=True).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            test_ds = self.strategy.experimental_distribute_dataset(test_ds)
            return train_ds, test_ds, self.channels

        elif type == 'cifar10':
            self.channels = 3
            (train_images, train_labels), (test_images, test_labels) = self.cifar10
            train_images, test_images = train_images.reshape(
                -1, 32, 32, 3), test_images.reshape(-1, 32, 32, 3)
            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images)).shuffle(10000).batch(self.batch_size, drop_remainder=True).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images)).batch(self.batch_size, drop_remainder=True).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            test_ds = self.strategy.experimental_distribute_dataset(test_ds)
            return train_ds, test_ds, self.channels

        elif type == 'fashion_mnist':
            self.num_classes = 10
            self.channels = 1
            (train_images, train_labels), (test_images,
                                           test_labels) = self.fashion_mnist
            train_images, test_images = train_images.reshape(
                -1, 28, 28, 1), test_images.reshape(-1, 28, 28, 1)

            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images)).shuffle(10000).batch(self.batch_size, drop_remainder=True).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images)).batch(self.batch_size, drop_remainder=True).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            test_ds = self.strategy.experimental_distribute_dataset(test_ds)
            return train_ds, test_ds, self.channels

        elif type == 'cifar100':
            self.channels = 3
            (train_images, train_labels), (test_images, test_labels) = self.cifar100
            train_images, test_images = train_images.reshape(
                -1, 32, 32, 3), test_images.reshape(-1, 32, 32, 3)
            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images)).shuffle(10000).batch(self.batch_size, drop_remainder=True).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images)).batch(self.batch_size, drop_remainder=True).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            test_ds = self.strategy.experimental_distribute_dataset(test_ds)
            return train_ds, test_ds, self.channels


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: ", strategy.num_replicas_in_sync)
    dataset = Dataset(strategy=strategy)
    for type in dataset.data_types:
        train_ds, test_ds,  channels = dataset.load_data(
            type)
        for image in train_ds:
            print(image)
            break
    # dataset = SkinCancerDataset()
