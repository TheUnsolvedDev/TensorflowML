import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *

STRIDES = np.array(YOLO_STRIDES)
ANCHORS = (np.array(YOLO_ANCHORS).T/STRIDES).T


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_not(training)
        return super().call(x, training=training)


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(
            ((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides, padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(
        0.0005), kernel_initializer=tf.random_normal_initializer(stddev=0.01), bias_initializer=tf.constant_initializer(0.1))(input_layer)
    if bn:
        conv = BatchNormalization()(conv)

    if activate:
        conv = tf.keras.layers.LeakyReLU(alpha=0.1)(conv)

    return conv


def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    for i in range(2):
        conv = convolutional(input_layer, filters_shape=(
            1, 1, input_channel, filter_num1), activate=True, bn=True)
        conv = convolutional(conv, filters_shape=(
            3, 3, filter_num1, filter_num2), activate=True, bn=True)
        if i == 0:
            input_layer = convolutional(short_cut, filters_shape=(
                1, 1, input_channel, filter_num2), activate=True, bn=True)
        input_layer = tf.keras.layers.Add()([input_layer, conv])
        input_layer = tf.keras.layers.LeakyReLU(alpha=0.1)(input_layer)
    return input_layer


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1]*2, input_layer.shape[2]*2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def darknet53(input_data):
    input_data = convolutional(input_data, filters_shape=(
        3, 3, 3, 32), activate=True, bn=True)
    input_data = convolutional(input_data, filters_shape=(
        3, 3, 32, 64), downsample=True, activate=True, bn=True)

    for i in range(1):
        input_data = residual_block(input_data, 64, 32, 64)

    input_data = convolutional(input_data, filters_shape=(
        3, 3, 64, 128), downsample=True, activate=True, bn=True)

    for i in range(2):
        input_data = residual_block(input_data, 128, 64, 128)

    input_data = convolutional(input_data, filters_shape=(
        3, 3, 128, 256), downsample=True, activate=True, bn=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data

    input_data = convolutional(input_data, filters_shape=(
        3, 3, 256, 512), downsample=True, activate=True, bn=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data

    input_data = convolutional(input_data, filters_shape=(
        3, 3, 512, 1024), downsample=True, activate=True, bn=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data


def darknet19_tiny(input_data):
    input_data = convolutional(input_data, filters_shape=(
        3, 3, 3, 16), activate=True, bn=True)
    input_data = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=2, padding='same')(input_data)
    input_data = convolutional(input_data, filters_shape=(
        3, 3, 16, 32), activate=True, bn=True)
    input_data = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=2, padding='same')(input_data)
    input_data = convolutional(input_data, filters_shape=(
        3, 3, 32, 64), activate=True, bn=True)
    input_data = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=2, padding='same')(input_data)
    input_data = convolutional(input_data, filters_shape=(
        3, 3, 64, 128), activate=True, bn=True)
    input_data = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=2, padding='same')(input_data)
    input_data = convolutional(input_data, filters_shape=(
        3, 3, 128, 256), activate=True, bn=True)
    route_1 = input_data
    input_data = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=2, padding='same')(input_data)
    input_data = convolutional(input_data, filters_shape=(
        3, 3, 256, 512), activate=True, bn=True)
    input_data = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=1, padding='same')(input_data)
    input_data = convolutional(input_data, filters_shape=(
        3, 3, 512, 512), activate=True, bn=True)
    return route_1, input_data


def YOLOv3_model(input_layer, NUM_CLASS):
    route_1, route_2, conv = darknet53(input_layer)

    conv = convolutional(conv, filters_shape=(
        1, 1, 1024, 512), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        3, 3, 512, 1024), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        1, 1, 1024, 512), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        3, 3, 512, 1024), activate=True, bn=True)

    conv_lobj_branch = convolutional(conv, filters_shape=(
        1, 1, 1024, 512), activate=True, bn=True)
    conv_lbbox = convolutional(
        conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, filters_shape=(
        1, 1, 512, 256), activate=True, bn=True)
    conv = tf.keras.layers.UpSampling2D(2)(conv)
    conv = tf.keras.layers.Concatenate()([conv, route_2])

    conv = convolutional(conv, filters_shape=(
        1, 1, 768, 256), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        3, 3, 256, 512), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        1, 1, 512, 256), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        3, 3, 256, 512), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        1, 1, 512, 256), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        3, 3, 256, 512), activate=True, bn=True)

    conv_mobj_branch = convolutional(conv, filters_shape=(
        1, 1, 512, 256), activate=True, bn=True)
    conv_mbbox = convolutional(
        conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(conv, filters_shape=(
        1, 1, 256, 128), activate=True, bn=True)
    conv = tf.keras.layers.UpSampling2D(2)(conv)
    conv = tf.keras.layers.Concatenate()([conv, route_1])

    conv = convolutional(conv, filters_shape=(
        1, 1, 384, 128), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        3, 3, 128, 256), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        1, 1, 256, 128), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        3, 3, 128, 256), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        1, 1, 256, 128), activate=True, bn=True)
    conv = convolutional(conv, filters_shape=(
        3, 3, 128, 256), activate=True, bn=True)

    conv_sobj_branch = convolutional(conv, filters_shape=(
        1, 1, 256, 128), activate=True, bn=True)
    conv_sbbox = convolutional(
        conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    return conv_lbbox, conv_mbbox, conv_sbbox

# def decode(conv_output, NUM_CLASS, i=0):
#     conv_shape = tf.shape(conv_output)
#     batch_size = conv_shape[0]
#     output_size = conv_shape[1]

#     conv_output = tf.reshape(
#         conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

#     conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
#     conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
#     conv_raw_conf = conv_output[:, :, :, :, 4:5]
#     conv_raw_prob = conv_output[:, :, :, :, 5:]
    
#     y = tf.range(output_size, dtype=tf.float32)
#     y = tf.expand_dims(y, -1)
#     y = tf.tile(y, [1, output_size])

#     x = tf.range(output_size, dtype=tf.float32)
#     x = tf.expand_dims(x, -1)
#     x = tf.tile(x, [output_size, 1])
    
#     xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
#     xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
#     xy_grid = tf.cast(xy_grid, tf.float32)
    
#     pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
#     pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    
#     pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
#     pred_conf = tf.sigmoid(conv_raw_conf)
#     pred_prob = tf.sigmoid(conv_raw_prob)
    
#     return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def YOLOv3_tiny_model(input_layer, NUM_CLASS):
    route_1, conv = darknet19_tiny(input_layer)
    
    conv = convolutional(conv,(1, 1, 1024, 256), activate=True, bn=True)
    conv_lobj_branch = convolutional(conv, (3, 3, 256, 512), activate=True, bn=True)
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)
    
    conv = convolutional(conv,(1, 1, 256, 128), activate=True, bn=True)
    conv = tf.keras.layers.UpSampling2D(2)(conv)
    conv = tf.keras.layers.Concatenate(axis = -1)([conv, route_1])

    conv = convolutional(conv,(1, 1, 128, 256), activate=True, bn=True)
    conv_mobj_branch = convolutional(conv, (3, 3, 128, 256), activate=True, bn=True)
    conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 256, 3*(NUM_CLASS + 5)), activate=False, bn=False)

    return conv_lbbox, conv_mbbox

class DecodeLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, strides, anchors, i=0):
        super(DecodeLayer, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.anchors = anchors
        self.i = i

    def call(self, conv_output):
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(
            conv_output, (batch_size, output_size, output_size, 3, 5 + self.num_classes)
        )

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.range(output_size, dtype=tf.float32)
        y = tf.reshape(y, [output_size, 1])  
        y = tf.tile(y, [1, output_size])  

        x = tf.range(output_size, dtype=tf.float32)
        x = tf.reshape(x, [1, output_size])  
        x = tf.tile(x, [output_size, 1])  

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(
            xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1]
        )
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * self.strides[self.i]
        pred_wh = (tf.exp(conv_raw_dwdh) * self.anchors[self.i]) * self.strides[self.i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def YOLOv3(img_size=IMAGE_SIZE, num_classes=len(COCO_LABELS), training=False, tiny=True):
    input_layer = tf.keras.layers.Input((img_size, img_size, 3))

    if tiny:
        conv_tensors = YOLOv3_tiny_model(input_layer, num_classes)
    else:
        conv_tensors = YOLOv3_model(input_layer, num_classes)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        decode_layer = DecodeLayer(num_classes, STRIDES, ANCHORS, i)
        pred_tensor = decode_layer(conv_tensor)
        if training:
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    return tf.keras.Model(input_layer, output_tensors, name='YOLOv3' if not tiny else 'YOLOv3-tiny')


if __name__ == '__main__':
    model = YOLOv3(tiny=True,training=True)
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
