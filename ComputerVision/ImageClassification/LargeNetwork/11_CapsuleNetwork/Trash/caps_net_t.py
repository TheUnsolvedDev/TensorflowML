import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU number')
args = parser.parse_args()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[args.gpu], 'GPU')


callbacks = [
    tf.keras.callbacks.CSVLogger('./log.csv'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model_capsnet.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
]


def process_images(image, label):
    image = tf.expand_dims(image, axis=-1)
    image /= 255.0
    label = tf.one_hot(label,depth = 10)
    return (image, label), (label, image)


class Length(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.math.sqrt(tf.reduce_sum(tf.square(inputs), -1) + 1e-10)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = tf.math.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            mask = tf.one_hot(indices=tf.argmax(
                x, 1), depth=x.get_shape().as_list()[1])
        masked = tf.keras.backend.batch_flatten(
            inputs * tf.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / \
        tf.math.sqrt(s_squared_norm + 1e-9)
    return scale * vectors


class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(
            input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_tiled = tf.expand_dims(inputs_tiled, 4)

        inputs_hat = tf.map_fn(lambda x: tf.matmul(
            self.W, x), elems=inputs_tiled)
        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule,
                            self.input_num_capsule, 1, 1])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            c = tf.keras.layers.Softmax(axis=1)(b)
            outputs = tf.multiply(c, inputs_hat)
            outputs = tf.reduce_sum(outputs, axis=2, keepdims=True)
            outputs = squash(outputs, axis=-2)

            if i < self.routings - 1:
                outputs_tiled = tf.tile(
                    outputs, [1, 1, self.input_num_capsule, 1, 1])
                agreement = tf.matmul(
                    inputs_hat, outputs_tiled, transpose_a=True)
                b = tf.add(b, agreement)
        outputs = tf.squeeze(outputs, [2, 4])
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = tf.keras.layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                                    name='primarycap_conv2d')(inputs)
    outputs = tf.keras.layers.Reshape(
        target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return tf.keras.layers.Lambda(squash, name='primarycap_squash')(outputs)


def margin_loss(y_true, y_pred):
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, 1))


def CapsNet(input_shape, n_class, routings):
    x = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(
        filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(
        conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)

    y = tf.keras.layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])
    masked = Mask()(digitcaps)

    # Shared Decoder model in training and prediction
    decoder = tf.keras.models.Sequential(name='decoder')
    decoder.add(tf.keras.layers.Dense(
        512, activation='relu', input_dim=16*n_class))
    decoder.add(tf.keras.layers.Dense(1024, activation='relu'))
    decoder.add(tf.keras.layers.Dense(
        np.prod(input_shape), activation='sigmoid'))
    decoder.add(tf.keras.layers.Reshape(
        target_shape=input_shape, name='out_recon'))

    train_model = tf.keras.models.Model(
        [x, y], [out_caps, decoder(masked_by_y)])
    eval_model = tf.keras.models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = tf.keras.layers.Input(shape=(n_class, 16))
    noised_digitcaps = tf.keras.layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = tf.keras.models.Model(
        [x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def train(model, train_data, validation_data):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.4],
                  metrics={'capsnet': 'accuracy'})

    model.fit(train_data, epochs=100,
              validation_data=validation_data,
              callbacks=callbacks)
    model.save_weights(args.save_dir + './model_capsnet.h5')
    return model


if __name__ == '__main__':
    model, eval_model, manipulate_model = CapsNet(input_shape=(28, 28, 1),
                                                  n_class=10,
                                                  routings=3)
    model.summary()
    tf.keras.utils.plot_model(model, to_file=CapsNet.__name__+'.png')

    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.mnist.load_data()
    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_images.astype(np.float32), train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_images.astype(np.float32), test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices(
        (validation_images.astype(np.float32), validation_labels))

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
                .batch(batch_size=2048, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=train_ds_size)
               .batch(batch_size=2048, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=train_ds_size)
                     .batch(batch_size=2048, drop_remainder=True))
    model = train(model, train_ds, validation_ds)
