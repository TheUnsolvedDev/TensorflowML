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
        filepath='model_mobile_net_v3.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def activation(x, at):
    if at == "RE":
        # ReLU6
        x = tf.keras.activations.relu(x, max_value=6)
    else:
        # Hard swish
        x = x * tf.keras.activations.relu(x, max_value=6) / 6

    return x


def _squeeze(x):
    x_copy = x
    channel = x.shape[-1]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(channel, activation="relu")(x)
    x = tf.keras.layers.Dense(channel, activation="hard_sigmoid")(x)
    x = tf.keras.layers.Reshape((1, 1, channel))(x)
    x = tf.keras.layers.Multiply()([x_copy, x])
    return x


def bneck(x, filters, kernel, expansion, strides, squeeze, at):
    x_copy = x

    input_shape = x.shape
    tchannel = int(expansion)
    cchannel = int(filters)

    r = strides == 1 and input_shape[3] == filters

    # Expansion convolution
    exp_x = tf.keras.layers.Conv2D(
        tchannel, (1, 1), padding="same", strides=(1, 1))(x)
    exp_x = tf.keras.layers.BatchNormalization(axis=-1)(exp_x)
    exp_x = activation(exp_x, at)

    # Depthwise convolution
    dep_x = tf.keras.layers.DepthwiseConv2D(
        kernel, strides=(strides, strides), depth_multiplier=1, padding="same"
    )(exp_x)
    dep_x = tf.keras.layers.BatchNormalization(axis=-1)(dep_x)
    dep_x = activation(dep_x, at)

    # Squeeze
    if squeeze:
        dep_x = _squeeze(dep_x)

    # Projection convolution
    pro_x = tf.keras.layers.Conv2D(
        cchannel, (1, 1), strides=(1, 1), padding="same")(dep_x)
    pro_x = tf.keras.layers.BatchNormalization(axis=-1)(pro_x)

    x = pro_x

    if r:
        x = tf.keras.layers.Add()([pro_x, x_copy])

    return x, exp_x, dep_x, pro_x


def Mobilenet():
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(inputs)
    x = tf.keras.layers.ReLU(6, name='conv1_relu')(x)

    x = activation(x, "HS")

    x, _, _, _ = bneck(
        x, 16, (3, 3), expansion=16, strides=1, squeeze=False, at="RE"
    )

    # 1/4
    x, _, _, _ = bneck(
        x, 24, (3, 3), expansion=64, strides=2, squeeze=False, at="RE"
    )
    x, _, _, _ = bneck(
        x, 24, (3, 3), expansion=72, strides=1, squeeze=False, at="RE"
    )

    # 1/8
    x, _, _, _ = bneck(
        x, 40, (5, 5), expansion=72, strides=2, squeeze=True, at="RE"
    )
    x, _, _, _ = bneck(
        x, 40, (5, 5), expansion=120, strides=1, squeeze=True, at="RE"
    )
    x_8, _, _, _ = bneck(
        x, 40, (5, 5), expansion=120, strides=1, squeeze=True, at="RE"
    )

    # 1/16
    x, _, _, _ = bneck(
        x_8, 80, (3, 3), expansion=240, strides=2, squeeze=False, at="HS"
    )
    x, _, _, _ = bneck(
        x, 80, (3, 3), expansion=200, strides=1, squeeze=False, at="HS"
    )
    x, _, _, _ = bneck(
        x, 80, (3, 3), expansion=184, strides=1, squeeze=False, at="HS"
    )
    x, _, _, _ = bneck(
        x, 80, (3, 3), expansion=184, strides=1, squeeze=False, at="HS"
    )
    x, _, _, _ = bneck(
        x, 112, (3, 3), expansion=480, strides=1, squeeze=True, at="HS"
    )
    x_16, _, _, _ = bneck(
        x, 112, (3, 3), expansion=672, strides=1, squeeze=True, at="HS"
    )
    # 1/32
    # 13th bottleneck block (C4) https://arxiv.org/pdf/1905.02244v4.pdf p.7
    x, _, _, _ = bneck(
        x_8, 160, (5, 5), expansion=672, strides=2, squeeze=True, at="HS"
    )
    x, _, _, _ = bneck(
        x, 160, (5, 5), expansion=960, strides=1, squeeze=True, at="HS"
    )
    x, _, _, _ = bneck(
        x, 160, (5, 5), expansion=960, strides=1, squeeze=True, at="HS"
    )

    # Layer immediatly before pooling (C5) https://arxiv.org/pdf/1905.02244v4.pdf p.7
    x = tf.keras.layers.Conv2D(960, (1, 1), strides=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = activation(x, at="HS")

    # Pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 960))(x)

    x = tf.keras.layers.Conv2D(1280, (1, 1), padding="same")(x)
    x = activation(x, "HS")
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
