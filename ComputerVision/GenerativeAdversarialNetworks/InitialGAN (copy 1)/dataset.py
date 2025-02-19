import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from config import *


class Dataset:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist.load_data()
        self.cifar10 = tf.keras.datasets.cifar10.load_data()
        self.fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
        self.cifar100 = tf.keras.datasets.cifar100.load_data()
        self.data_types = ['cifar10', 'fashion_mnist',
                           'mnist',  'cifar100']
        self.batch_size = BATCH_SIZE
        self.img_shape = IMAGE_SIZE

    def process_images(self, image):
        # image = tf.image.per_image_standardization(image)
        image = tf.image.resize(image, self.img_shape)
        return image

    def load_data(self, type='mnist'):
        if type == 'mnist':
            self.channels = 1
            (train_images, train_labels), (test_images, test_labels) = self.mnist
            train_images, test_images = train_images.reshape(
                -1, 28, 28, 1), test_images.reshape(-1, 28, 28, 1)
            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images)).shuffle(10000).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images)).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)

            return train_ds, test_ds, self.channels

        elif type == 'cifar10':
            self.channels = 3
            (train_images, train_labels), (test_images, test_labels) = self.cifar10
            train_images, test_images = train_images.reshape(
                -1, 32, 32, 3), test_images.reshape(-1, 32, 32, 3)
            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images)).shuffle(10000).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images)).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)

            return train_ds, test_ds, self.channels

        elif type == 'fashion_mnist':
            self.num_classes = 10
            self.channels = 1
            (train_images, train_labels), (test_images,
                                           test_labels) = self.fashion_mnist
            train_images, test_images = train_images.reshape(
                -1, 28, 28, 1), test_images.reshape(-1, 28, 28, 1)

            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images)).shuffle(10000).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images)).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)

            return train_ds, test_ds, self.channels

        elif type == 'cifar100':
            self.channels = 3
            (train_images, train_labels), (test_images, test_labels) = self.cifar100
            train_images, test_images = train_images.reshape(
                -1, 32, 32, 3), test_images.reshape(-1, 32, 32, 3)
            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images)).shuffle(10000).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images)).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)

            return train_ds, test_ds, self.channels


if __name__ == "__main__":
    dataset = Dataset()
    for type in dataset.data_types:
        train_ds, test_ds,  channels = dataset.load_data(
            type)
        for image in train_ds.take(1):
            print(image.shape)
            break
    # dataset = SkinCancerDataset()
