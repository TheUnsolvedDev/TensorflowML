import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import os

from config import *


class Dataset:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist.load_data()
        self.cifar10 = tf.keras.datasets.cifar10.load_data()
        self.fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
        self.cifar100 = tf.keras.datasets.cifar100.load_data()
        self.data_types = ['cifar10', 'fashion_mnist', 'mnist',  'cifar100']
        self.batch_size = BATCH_SIZE
        self.img_shape = [INPUT_SIZE[0], INPUT_SIZE[1]]

    def process_images(self, image, label):
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize(image, self.img_shape)
        return image, label

    def load_data(self, type='mnist'):
        if type == 'mnist':
            self.num_classes = 10
            self.channels = 1
            (train_images, train_labels), (test_images, test_labels) = self.mnist
            train_images, test_images = train_images.reshape(
                -1, 28, 28, 1), test_images.reshape(-1, 28, 28, 1)

        elif type == 'cifar10':
            self.num_classes = 10
            self.channels = 3
            (train_images, train_labels), (test_images, test_labels) = self.cifar10
            train_images, test_images = train_images.reshape(
                -1, 32, 32, 3), test_images.reshape(-1, 32, 32, 3)

        elif type == 'fashion_mnist':
            self.num_classes = 10
            self.channels = 1
            (train_images, train_labels), (test_images,
                                           test_labels) = self.fashion_mnist
            train_images, test_images = train_images.reshape(
                -1, 28, 28, 1), test_images.reshape(-1, 28, 28, 1)

        elif type == 'cifar100':
            self.num_classes = 100
            self.channels = 3
            (train_images, train_labels), (test_images, test_labels) = self.cifar100
            train_images, test_images = train_images.reshape(
                -1, 32, 32, 3), test_images.reshape(-1, 32, 32, 3)

        train_labels, test_labels = tf.one_hot(
            train_labels, depth=self.num_classes), tf.one_hot(test_labels, depth=self.num_classes)
        train_labels, test_labels = np.squeeze(
            train_labels), np.squeeze(test_labels)
        train_images, validation_images, train_labels, validation_labels = train_images[
            :int(len(train_labels)*0.8)], train_images[int(len(train_labels)*0.8):], train_labels[:int(len(train_labels)*0.8)], train_labels[int(len(train_labels)*0.8):]

        train_ds = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels)).shuffle(10000).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)
        validation_ds = tf.data.Dataset.from_tensor_slices(
            (validation_images, validation_labels)).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)
        test_ds = tf.data.Dataset.from_tensor_slices(
            (test_images, test_labels)).batch(self.batch_size).map(self.process_images).prefetch(tf.data.AUTOTUNE)

        return train_ds, validation_ds, test_ds, self.num_classes, self.channels


if __name__ == "__main__":
    dataset = Dataset()
    for type in dataset.data_types:
        train_ds, validation_ds, test_ds, num_classes, channels = dataset.load_data(
            type)
        for image, label in validation_ds.take(1):
            print(image.shape)
            print(label.shape)
            break