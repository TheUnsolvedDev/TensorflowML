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
parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU number')
parser.add_argument('--name', type=str, default='vangogh2photo',
                    help='Name of the Dataset to select')
args = parser.parse_args()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if args.gpu >= 0 and args.gpu <= 3:
    tf.config.experimental.set_visible_devices(
        physical_devices[args.gpu], 'GPU')
else:
    for device in physical_devices[1:]:
        tf.config.experimental.set_visible_devices(device, 'GPU')
        tf.config.experimental.set_memory_growth(device, enable=True)


class TensorLookup:
    def __init__(self, keys, strings):
        self.keys = keys
        self.strings = strings
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self.keys, tf.range(tf.shape(self.keys)[0])),
            default_value=-1)

    def lookup(self, key):
        index = self.table.lookup(key)
        return self.strings[index]


class tensorflow_pipeline:
    def __init__(self, batch_size=128):
        data = pd.read_csv(
            'Datasets/list_attr_celeba.csv')
        ATTRIBUTES = ['Black_Hair', 'Blond_Hair',
                      'Brown_Hair', 'Male', 'Young']
        for attr in data.columns[1:]:
            if attr not in ATTRIBUTES:
                data = data.drop([attr], axis=1)
        data = np.array(data.replace(-1, 0))
        self.batch_size = batch_size
        keys = tf.convert_to_tensor(data[:, 0], dtype=tf.string)
        values = tf.convert_to_tensor(data[:, 1:], dtype=tf.int32)
        # self.table = tf.lookup.StaticHashTable(
        #     tf.lookup.KeyValueTensorInitializer(keys, values),
        #     default_value=-1)
        self.table = TensorLookup(keys, values)
        self.base_path = './'
        self.files = tf.data.Dataset.list_files(
            self.base_path+'*')

    def normalize(self, image):
        image = tf.cast(image, dtype=tf.float32)
        image = (image / 127.5) - 1
        return image

    def resize(self, image, size):
        h, w = size
        image = tf.image.resize(image, [h, w])
        return image

    def process_path(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img)
        return img

    def preprocess_data(self, file_path):
        image = self.process_path(file_path)
        image = self.resize(image, (128, 128))
        image = self.normalize(image)
        return image

    def get_label(self, path):
        path = tf.strings.split(path, sep='/')[-1]
        return self.table.lookup(path)

    def parse_func(self, files):
        img = []
        label = []
        new_targets = []
        for i in range(len(files)):
            label.append(self.get_label(files[i]))
            new_targets.append(self.get_label(files[i]))
            img.append(self.preprocess_data(files[i]))
        random.shuffle(new_targets)
        return img, label, new_targets

    def train_gen(self):
        for img, label, new_target in self.train_img:
            yield img, label, new_target

    def load_images_gen(self):
        train_files = self.files
        train_ds_batch = train_files.batch(
            self.batch_size, drop_remainder=True)
        self.train_img = train_ds_batch.map(
            lambda x: self.parse_func(x))

        train_gen = tf.data.Dataset.from_generator(
            generator=self.train_gen,
            output_signature=(
                tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 5), dtype=tf.int32),
                tf.TensorSpec(shape=(None, 5), dtype=tf.int32))).prefetch(tf.data.AUTOTUNE)
        return train_gen


def create_dataset():
    pipe = tensorflow_pipeline()
    gen = pipe.load_images_gen()
    return gen


if __name__ == '__main__':
    data = create_dataset()
    for ind in tqdm.tqdm(range(len(data))):
        a, b, c = data.__getitem__(ind)
        print(ind,a.shape, b, c)
        break
    # print(100010010)
    # pipe = tensorflow_pipeline()
    # gen = pipe.load_images_gen()
    # for x, y, z in gen.take(100):
    #     print(x.shape, y.shape, z.shape)
