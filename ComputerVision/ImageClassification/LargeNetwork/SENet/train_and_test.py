import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import argparse

from config import *
from dataset import *
from model import *


def main():
    parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number')
    parser.add_argument('--type', type=str, default='cifar10',
                        help='Dataset type', choices=['cifar10', 'fashion_mnist', 'mnist', 'cifar100'])
    args = parser.parse_args()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if int(args.gpu) != -1:
        tf.config.experimental.set_visible_devices(
            physical_devices[args.gpu], 'GPU')

    dataset = Dataset()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model_squeze_excitation_resnet50_' + args.type + '.weights.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs', histogram_freq=1, write_graph=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]

    train_ds, validation_ds, test_ds, num_classes, channels = dataset.load_data(
        args.type)
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = squeze_excitation_resnet50_model(input_shape=(
            INPUT_SIZE[0], INPUT_SIZE[1], channels), num_classes=num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    model.summary(expand_nested=True)
    tf.keras.utils.plot_model(
        model, to_file=squeze_excitation_resnet50_model.__name__+'.png')
    model.fit(train_ds, validation_data=validation_ds,
              epochs=EPOCHS, callbacks=callbacks)
    model.evaluate(test_ds)


if __name__ == '__main__':
    main()
