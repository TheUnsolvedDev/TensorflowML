import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import argparse

from config import *
from dataset import *
from model import *


def main():
    model_fn = densenet_model_169
    parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number')
    parser.add_argument('--type', type=str, default='cifar10',
                        help='Dataset type', choices=['cifar10', 'fashion_mnist',
                           'mnist',  'cifar100', 'skin_cancer', 'cassava_leaf_disease', 'chest_xray', 'crop_disease'])
    args = parser.parse_args()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    if int(args.gpu) != -1:
        tf.config.experimental.set_visible_devices(
            physical_devices[args.gpu], 'GPU')

    dataset = Dataset()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=f'{model_fn.__name__}_' + args.type + '.weights.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs_{args.type}_{model_fn.__name__}', histogram_freq=1, write_graph=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]

    train_ds, validation_ds, test_ds, num_classes, channels = dataset.load_data(
        args.type)
    strategy = tf.distribute.MirroredStrategy()
    print(f'Training on dataset {args.type} with {strategy.num_replicas_in_sync} devices')

    with strategy.scope():
        model = model_fn(input_shape=(
            INPUT_SIZE[0], INPUT_SIZE[1], channels), num_classes=num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    model.summary(expand_nested=True)
    tf.keras.utils.plot_model(
        model, to_file=model_fn.__name__+'.png',show_shapes=True, show_layer_names=True)
    model.fit(train_ds, validation_data=validation_ds,
              epochs=EPOCHS, callbacks=callbacks)
    model.evaluate(test_ds)


if __name__ == '__main__':
    main()
