import os
import numpy as np
import functools
import matplotlib.pyplot as plt
import silence_tensorflow.auto
import tensorflow as tf
import gdown
import zipfile

from config import *

def log2(x):
    return int(np.log2(x))


def resize_image(res, image):
    image = tf.image.resize(
        image, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image


class Dataset:
    def __init__(self):
        os.makedirs("celeba_gan", exist_ok=True)
        url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
        output = "celeba_gan/data.zip"
        # gdown.download(url, output, quiet=False)

        self.batch_sizes = {2: 16, 3: 16, 4: 16,
                            5: 16, 6: 16, 7: 8, 8: 4, 9: 2, 10: 1}
        self.train_step_ratio = {
            k: self.batch_sizes[2] / v for k, v in self.batch_sizes.items()}
        with zipfile.ZipFile("celeba_gan/data.zip", "r") as zipobj:
            zipobj.extractall("celeba_gan")

        self.ds_train = tf.keras.utils.image_dataset_from_directory(
            "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=32
        )

    def create_dataloader(self, res):
        batch_size = self.batch_sizes[log2(res)]
        dl = self.ds_train.map(functools.partial(
            resize_image, res), num_parallel_calls=tf.data.AUTOTUNE).unbatch()
        dl = dl.shuffle(200).batch(
            batch_size, drop_remainder=True).prefetch(1).repeat()
        return dl


def plot_images(images, log2_res, fname=""):
    scales = {2: 0.5, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}
    scale = scales[log2_res]

    grid_col = min(images.shape[0], int(32 // scale))
    grid_row = 1

    f, axarr = plt.subplots(
        grid_row, grid_col, figsize=(grid_col * scale, grid_row * scale)
    )

    for row in range(grid_row):
        ax = axarr if grid_row == 1 else axarr[row]
        for col in range(grid_col):
            ax[col].imshow(images[row * grid_col + col])
            ax[col].axis("off")
    plt.show()
    if fname:
        f.savefig(fname)


if __name__ == "__main__":
    onj = Dataset()
    loader = onj.create_dataloader(128)
    images = next(iter(loader))
    plot_images(images, log2(128))
