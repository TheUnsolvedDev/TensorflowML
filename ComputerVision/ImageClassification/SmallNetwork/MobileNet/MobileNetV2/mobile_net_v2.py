import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse

parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU number')
args = parser.parse_args()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[args.gpu], 'GPU')
# tf.config.experimental.set_memory_growth(physical_devices[args.gpu], True)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_mobile_net_v2.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def expansion_block(x, t, filters, block_id):
    prefix = 'block_{}_'.format(block_id)
    total_filters = t*filters
    x = tf.keras.layers.Conv2D(total_filters, 1, padding='same',
                               use_bias=False, name=prefix + 'expand')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + 'expand_bn')(x)
    x = tf.keras.layers.ReLU(6, name=prefix + 'expand_relu')(x)
    return x


def depthwise_block(x, stride, block_id):
    prefix = 'block_{}_'.format(block_id)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=(stride, stride), padding='same',
                                        use_bias=False, name=prefix + 'depthwise_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + 'dw_bn')(x)
    x = tf.keras.layers.ReLU(6, name=prefix + 'dw_relu')(x)
    return x


def projection_block(x, out_channels, block_id):
    prefix = 'block_{}_'.format(block_id)
    x = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,   padding='same',
                               use_bias=False, name=prefix + 'compress')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + 'compress_bn')(x)
    return x


def Bottleneck(x, t, filters, out_channels, stride, block_id):
    y = expansion_block(x, t, filters, block_id)
    y = depthwise_block(y, stride, block_id)
    y = projection_block(y, out_channels, block_id)
    if y.shape[-1] == x.shape[-1]:
        y = tf.keras.layers.add([x, y])
    return y


def Mobilenet():
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(inputs)
    x = tf.keras.layers.ReLU(6, name='conv1_relu')(x)
    # 17 Bottlenecks
    x = depthwise_block(x, stride=1, block_id=1)
    x = projection_block(x, out_channels=16, block_id=1)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=24, stride=2, block_id=2)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=24, stride=1, block_id=3)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=32, stride=2, block_id=4)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=5)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=6)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=2, block_id=7)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=8)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=9)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=10)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=11)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=12)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=13)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=160, stride=2, block_id=14)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=15)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=16)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=320, stride=1, block_id=17)
    x = tf.keras.layers.Conv2D(
        filters=1280, kernel_size=1, padding='same', use_bias=False, name='last_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='last_bn')(x)
    x = tf.keras.layers.ReLU(6, name='last_relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pool')(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs, output)
    return model


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (64, 64))
    return image, label


if __name__ == '__main__':
    model = Mobilenet()
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file=Mobilenet.__name__+'.png', show_shapes=True)

    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.cifar10.load_data()
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices(
        (validation_images, validation_labels))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(
        validation_ds).numpy()
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)

    train_ds = (train_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=256, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=256, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=256, drop_remainder=True))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.optimizers.Adam(lr=0.01), metrics=['accuracy'])
    model.summary()
    model.fit(train_ds,
              epochs=50,
              validation_data=validation_ds,
              validation_freq=1,
              callbacks=callbacks)
    model.evaluate(test_ds)
