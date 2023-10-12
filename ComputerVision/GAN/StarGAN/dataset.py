import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tqdm
import random
from silence_tensorflow import silence_tensorflow
import argparse

from params import *

silence_tensorflow()


def process_data(path, true_label, random_label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    return img/255.0, true_label, random_label


class Datagen(tf.keras.utils.Sequence):
    def __init__(self, batch_size=BATCH_SIZE) -> None:
        self.batch_size = batch_size

        data = pd.read_csv(
            'Datasets/list_attr_celeba.csv')
        ATTRIBUTES = ['Black_Hair', 'Blond_Hair',
                      'Brown_Hair', 'Male', 'Young']
        for attr in data.columns[1:]:
            if attr not in ATTRIBUTES:
                data = data.drop([attr], axis=1)
        self.data = np.array(data.replace(-1, 0))
        self.image_paths = self.data[:, 0]
        self.labels = self.data[:, 1:]

    def __len__(self):
        return len(self.data)//self.batch_size

    def read_data(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
        img = (img/255.0, tf.float32)
        return img[0].numpy()

    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.data))
        idx = np.random.permutation(self.batch_size)
        batch_x = self.image_paths[low:high]
        batch_y = self.labels[low:high]
        batch_z = batch_y.copy()
        np.random.shuffle(batch_z)

        imgs, lbl, gen_lbl = [], [], []
        for i in idx:
            imgs.append(self.read_data(
                'Datasets/img_align_celeba/'+batch_x[i]))
            lbl.append(batch_y[i])
            gen_lbl.append(batch_z[i])

        return np.array(imgs).astype(np.float32), np.array(lbl).astype(np.float32), np.array(gen_lbl).astype(np.float32)


def dataset():
    data = pd.read_csv(
        'Datasets/list_attr_celeba.csv')
    ATTRIBUTES = ['Black_Hair', 'Blond_Hair',
                  'Brown_Hair', 'Male', 'Young']
    for attr in data.columns[1:]:
        if attr not in ATTRIBUTES:
            data = data.drop([attr], axis=1)
    data = np.array(data.replace(-1, 0))
    image_paths = 'Datasets/img_align_celeba/'+data[:, 0]
    labels = data[:, 1:].astype(np.float32)

    imgs = tf.data.Dataset.from_tensor_slices(image_paths)
    true_labels = tf.data.Dataset.from_tensor_slices(labels.copy())
    np.random.shuffle(labels)
    gen_label = tf.data.Dataset.from_tensor_slices(labels.copy())

    dataset = tf.data.Dataset.zip((imgs, true_labels, gen_label))
    dataset = dataset.map(
        lambda x, y, z: process_data(x, y, z)).batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':
    data = dataset()
    for i in data.take(1):
        print(i[0].shape, np.min(i[0]), np.max(i[0]))
        break

# if __name__ == '__main__':
#     pipe = Datagen()
#     a, b, d = pipe.__getitem__(0)
#     print(a)
