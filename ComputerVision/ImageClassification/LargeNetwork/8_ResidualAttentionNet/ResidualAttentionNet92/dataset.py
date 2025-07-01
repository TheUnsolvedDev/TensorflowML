import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2

from config import *


class SkinCancerDataset:
    def __init__(self, train_size=0.8):
        self.train_size = train_size
        self.dataset_path = DATASET_PATH+'skin_cancer/'
        self.df = pd.read_csv(self.dataset_path+'HAM10000_metadata.csv')
        self.classes = list(self.df['dx'].unique())
        self.classes_to_int = dict(zip(self.classes, range(len(self.classes))))
        self.int_to_classes = dict(zip(range(len(self.classes)), self.classes))
        self.image_locations = list(self.df['image_id'])
        self.labels = list(self.df['dx'])

        self.num_classes = len(self.classes)
        self.channels = 3

    def shuffle(self, data, labels):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        return data[indices], labels[indices]

    def prepare_dataset(self):
        self.image_locations = [
            self.dataset_path + 'skin_cancer_images/' + x + '.jpg' for x in self.image_locations]
        self.labels = [self.classes_to_int[x] for x in self.labels]
        self.image_locations, self.labels = np.array(
            self.image_locations), np.array(self.labels)
        self.image_locations, self.labels = self.shuffle(
            self.image_locations, self.labels)

        train_data = self.image_locations[:int(
            len(self.image_locations)*self.train_size)]
        train_labels = self.labels[:int(
            len(self.image_locations)*self.train_size)]
        test_data = self.image_locations[int(
            len(self.image_locations)*self.train_size):]
        test_labels = self.labels[int(
            len(self.image_locations)*self.train_size):]
        return train_data, train_labels, test_data, test_labels


class CassavaLeafDiseaseDataset:
    def __init__(self, train_size=0.8):
        self.train_size = train_size
        self.dataset_path = DATASET_PATH+'cassava_leaf_disease/'
        self.df = pd.read_csv(self.dataset_path+'merged.csv')
        self.classes = list(self.df['label'].unique())
        self.classes_to_int = dict(zip(self.classes, range(len(self.classes))))
        self.int_to_classes = dict(zip(range(len(self.classes)), self.classes))
        self.image_locations = list(self.df['image_id'])
        self.labels = list(self.df['label'])

        self.num_classes = len(self.classes)
        self.channels = 3

    def shuffle(self, data, labels):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        return data[indices], labels[indices]

    def prepare_dataset(self):
        self.image_locations = [
            self.dataset_path + 'train/' + x for x in self.image_locations]
        self.labels = [self.classes_to_int[x] for x in self.labels]
        self.image_locations, self.labels = np.array(
            self.image_locations), np.array(self.labels)
        self.image_locations, self.labels = self.shuffle(
            self.image_locations, self.labels)

        train_data = self.image_locations[:int(
            len(self.image_locations)*self.train_size)]
        train_labels = self.labels[:int(
            len(self.image_locations)*self.train_size)]
        test_data = self.image_locations[int(
            len(self.image_locations)*self.train_size):]
        test_labels = self.labels[int(
            len(self.image_locations)*self.train_size):]
        return train_data, train_labels, test_data, test_labels


class ChestXRayDataset:
    def __init__(self, train_size=0.8):
        self.train_size = train_size
        self.dataset_path = DATASET_PATH+'chest_x_ray/'
        self.train_data_dir = self.dataset_path+'train/'
        self.test_data_dir = self.dataset_path+'test/'
        self.val_data_dir = self.dataset_path+'val/'
        self.classes = ['NORMAL', 'PNEUMONIA', 'COVID19', 'TURBERCULOSIS']
        self.classes_to_int = dict(zip(self.classes, range(len(self.classes))))
        self.int_to_classes = dict(zip(range(len(self.classes)), self.classes))
        self.image_locations = []
        self.labels = []

        for folder in os.listdir(self.train_data_dir):
            for file in os.listdir(self.train_data_dir+folder):
                self.image_locations.append(
                    self.train_data_dir+folder+'/'+file)
                self.labels.append(self.classes_to_int[folder])
        for folder in os.listdir(self.test_data_dir):
            for file in os.listdir(self.test_data_dir+folder):
                self.image_locations.append(self.test_data_dir+folder+'/'+file)
                self.labels.append(self.classes_to_int[folder])
        for folder in os.listdir(self.val_data_dir):
            for file in os.listdir(self.val_data_dir+folder):
                self.image_locations.append(self.val_data_dir+folder+'/'+file)
                self.labels.append(self.classes_to_int[folder])

        self.num_classes = len(self.classes)
        self.channels = 3

    def shuffle(self, data, labels):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        return data[indices], labels[indices]

    def prepare_dataset(self):
        self.image_locations_old, self.labels_old = np.array(
            self.image_locations), np.array(self.labels)
        self.image_locations_old, self.labels_old = self.shuffle(
            self.image_locations_old, self.labels_old)

        self.image_locations, self.labels = [], []

        for ind, x in enumerate(self.image_locations_old):
            try:
                img = tf.io.read_file(x)
                img = tf.image.decode_jpeg(img, channels=3)
                self.image_locations.append(x)
                self.labels.append(self.labels_old[ind])
            except Exception as e:
                print(f'Error reading image: {x}, {e}')
                os.system(f'rm "{x}"')

        train_data = self.image_locations[:int(
            len(self.image_locations)*self.train_size)]
        train_labels = self.labels[:int(
            len(self.image_locations)*self.train_size)]
        test_data = self.image_locations[int(
            len(self.image_locations)*self.train_size):]
        test_labels = self.labels[int(
            len(self.image_locations)*self.train_size):]
        return train_data, train_labels, test_data, test_labels


class CropDiseaseDataset:
    def __init__(self, train_size=0.8):
        self.train_size = train_size
        self.dataset_path = DATASET_PATH+'crop_disease/'
        self.df = pd.read_csv(self.dataset_path+'train.csv')
        self.classes = list(self.df['label'].unique())
        self.classes_to_int = dict(zip(self.classes, range(len(self.classes))))
        self.int_to_classes = dict(zip(range(len(self.classes)), self.classes))
        self.image_locations = list(self.df['image_id'])
        self.labels = list(self.df['label'])

        self.num_classes = len(self.classes)
        self.channels = 3

    def shuffle(self, data, labels):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        return data[indices], labels[indices]

    def prepare_dataset(self):
        self.image_locations_old = [
            self.dataset_path + 'train_images/' + x for x in self.image_locations]
        self.labels_old = [self.classes_to_int[x] for x in self.labels]
        self.image_locations_old, self.labels_old = np.array(
            self.image_locations_old), np.array(self.labels_old)
        self.image_locations_old, self.labels_old = self.shuffle(
            self.image_locations_old, self.labels_old)

        self.image_locations, self.labels = [], []
        for ind, x in enumerate(self.image_locations_old):
            try:
                with open(x, 'rb') as f:
                    pass
                self.image_locations.append(x)
                self.labels.append(self.labels_old[ind])
            except Exception as e:
                pass

        train_data = self.image_locations[:int(
            len(self.image_locations)*self.train_size)]
        train_labels = self.labels[:int(
            len(self.image_locations)*self.train_size)]
        test_data = self.image_locations[int(
            len(self.image_locations)*self.train_size):]
        test_labels = self.labels[int(
            len(self.image_locations)*self.train_size):]
        return train_data, train_labels, test_data, test_labels


class Dataset:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist.load_data()
        self.cifar10 = tf.keras.datasets.cifar10.load_data()
        self.fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
        self.cifar100 = tf.keras.datasets.cifar100.load_data()
        self.data_types = ['cifar10', 'fashion_mnist',
                           'mnist',  'cifar100', 'skin_cancer', 'cassava_leaf_disease', 'chest_xray', 'crop_disease']
        self.batch_size = BATCH_SIZE
        self.img_shape = [INPUT_SIZE[0], INPUT_SIZE[1]]

    def process_images(self, image, label, decode=False):
        if decode:
            try:
                image = tf.io.read_file(image)
                image = tf.image.decode_jpeg(image, channels=3)
            except:
                pass
        # image = tf.image.per_image_standardization(image)
        image = tf.image.resize(image, self.img_shape)
        return image, label

    def load_data(self, type='mnist'):
        if type == 'mnist':
            self.num_classes = 10
            self.channels = 1
            (train_images, train_labels), (test_images, test_labels) = self.mnist
            train_images, test_images = train_images.reshape(
                -1, 28, 28, 1), test_images.reshape(-1, 28, 28, 1)
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

        elif type == 'cifar10':
            self.num_classes = 10
            self.channels = 3
            (train_images, train_labels), (test_images, test_labels) = self.cifar10
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

        elif type == 'fashion_mnist':
            self.num_classes = 10
            self.channels = 1
            (train_images, train_labels), (test_images,
                                           test_labels) = self.fashion_mnist
            train_images, test_images = train_images.reshape(
                -1, 28, 28, 1), test_images.reshape(-1, 28, 28, 1)
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

        elif type == 'skin_cancer':
            dataset = SkinCancerDataset()
            train_images, train_labels, test_images, test_labels = dataset.prepare_dataset()
            self.num_classes = dataset.num_classes
            self.channels = 3
            train_labels, test_labels = tf.one_hot(
                train_labels, depth=self.num_classes), tf.one_hot(test_labels, depth=self.num_classes)
            train_labels, test_labels = np.squeeze(
                train_labels), np.squeeze(test_labels)
            train_images, validation_images, train_labels, validation_labels = train_images[
                :int(len(train_labels)*0.8)], train_images[int(len(train_labels)*0.8):], train_labels[:int(len(train_labels)*0.8)], train_labels[int(len(train_labels)*0.8):]

            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images, train_labels)).shuffle(10000).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            validation_ds = tf.data.Dataset.from_tensor_slices(
                (validation_images, validation_labels)).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images, test_labels)).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

            return train_ds, validation_ds, test_ds, self.num_classes, self.channels

        elif type == 'cassava_leaf_disease':
            dataset = CassavaLeafDiseaseDataset()
            train_images, train_labels, test_images, test_labels = dataset.prepare_dataset()
            self.num_classes = dataset.num_classes
            self.channels = 3
            train_labels, test_labels = tf.one_hot(
                train_labels, depth=self.num_classes), tf.one_hot(test_labels, depth=self.num_classes)
            train_labels, test_labels = np.squeeze(
                train_labels), np.squeeze(test_labels)
            train_images, validation_images, train_labels, validation_labels = train_images[
                :int(len(train_labels)*0.8)], train_images[int(len(train_labels)*0.8):], train_labels[:int(len(train_labels)*0.8)], train_labels[int(len(train_labels)*0.8):]

            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images, train_labels)).shuffle(10000).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            validation_ds = tf.data.Dataset.from_tensor_slices(
                (validation_images, validation_labels)).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images, test_labels)).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            return train_ds, validation_ds, test_ds, self.num_classes, self.channels

        elif type == 'chest_xray':
            dataset = ChestXRayDataset()
            train_images, train_labels, test_images, test_labels = dataset.prepare_dataset()
            self.num_classes = dataset.num_classes
            self.channels = 3
            train_labels, test_labels = tf.one_hot(
                train_labels, depth=self.num_classes), tf.one_hot(test_labels, depth=self.num_classes)
            train_labels, test_labels = np.squeeze(
                train_labels), np.squeeze(test_labels)
            train_images, validation_images, train_labels, validation_labels = train_images[
                :int(len(train_labels)*0.8)], train_images[int(len(train_labels)*0.8):], train_labels[:int(len(train_labels)*0.8)], train_labels[int(len(train_labels)*0.8):]

            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images, train_labels)).shuffle(10000).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            validation_ds = tf.data.Dataset.from_tensor_slices(
                (validation_images, validation_labels)).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images, test_labels)).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            return train_ds, validation_ds, test_ds, self.num_classes, self.channels

        elif type == 'crop_disease':
            dataset = CropDiseaseDataset()
            train_images, train_labels, test_images, test_labels = dataset.prepare_dataset()
            self.num_classes = dataset.num_classes
            self.channels = 3

            train_labels, test_labels = tf.one_hot(
                train_labels, depth=self.num_classes), tf.one_hot(test_labels, depth=self.num_classes)
            train_labels, test_labels = np.squeeze(
                train_labels), np.squeeze(test_labels)
            train_images, validation_images, train_labels, validation_labels = train_images[
                :int(len(train_labels)*0.8)], train_images[int(len(train_labels)*0.8):], train_labels[:int(len(train_labels)*0.8)], train_labels[int(len(train_labels)*0.8):]

            train_ds = tf.data.Dataset.from_tensor_slices(
                (train_images, train_labels)).shuffle(10000).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            validation_ds = tf.data.Dataset.from_tensor_slices(
                (validation_images, validation_labels)).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_tensor_slices(
                (test_images, test_labels)).map(lambda x, y: self.process_images(x, y, decode=True)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            return train_ds, validation_ds, test_ds, self.num_classes, self.channels


if __name__ == "__main__":
    dataset = Dataset()
    for type in dataset.data_types:
        train_ds, validation_ds, test_ds, num_classes, channels = dataset.load_data(
            type)
        for image, label in validation_ds:
            print(image.shape, label.shape)

    # dataset = SkinCancerDataset()
    # dataset = CassavaLeafDiseaseDataset()
    # train_images, train_labels, test_images, test_labels = dataset.prepare_dataset()
    # print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
