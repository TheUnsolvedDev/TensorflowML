import tensorflow as tf
from utils import *
from config import *


import silence_tensorflow.auto
import tensorflow as tf
import argparse
import os

from dataset import *
from model import *


def conv_block(x, filters, kernel_size, strides):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    return x

def inception_resnet_block(x, filters):
    shortcut = x

    branch1 = conv_block(x, filters, 1, 1)

    branch2 = conv_block(x, filters, 1, 1)
    branch2 = conv_block(branch2, filters, 3, 1)

    branch3 = conv_block(x, filters, 1, 1)
    branch3 = conv_block(branch3, filters, 3, 1)
    branch3 = conv_block(branch3, filters, 3, 1)

    mixed = tf.keras.layers.Concatenate()([branch1, branch2, branch3])
    mixed = conv_block(mixed, filters, 1, 1)

    if shortcut.shape[-1] != mixed.shape[-1]:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([shortcut, mixed])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_classifier():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomBrightness(factor=0.1)(x)
    x = tf.keras.layers.RandomContrast(factor=0.1)(x)
    x = conv_block(x, 32, 3, 2)
    x = conv_block(x, 32, 3, 1)
    x = conv_block(x, 64, 3, 1)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    for _ in range(3):
        x = inception_resnet_block(x, 64)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    for _ in range(3):
        x = inception_resnet_block(x, 128)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name="Classifier_InceptionResNet")
    model.summary()
    return model

def build_yolo_v1(backbone_weights=None, freeze_backbone=False):
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomBrightness(factor=0.1)(x)
    x = tf.keras.layers.RandomContrast(factor=0.1)(x)

    x = conv_block(x, 32, 3, 2)
    x = conv_block(x, 32, 3, 1)
    x = conv_block(x, 64, 3, 1)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    for _ in range(3):
        x = inception_resnet_block(x, 64)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    for _ in range(3):
        x = inception_resnet_block(x, 128)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    backbone = tf.keras.Model(inputs, x, name="inception_resnet_backbone")
    if backbone_weights:
        print(f"Loading backbone weights from: {backbone_weights}")
        backbone.load_weights(backbone_weights)
    if freeze_backbone:
        print("Freezing backbone layers.")
        backbone.trainable = False

    x = backbone(inputs)

    x = conv_block(x, 256, 3, 1)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_units = S * S * (B * 5 + NUM_CLASSES)
    x = tf.keras.layers.Dense(output_units)(x)
    outputs = tf.keras.layers.Reshape((S, S, B * 5 + NUM_CLASSES))(x)
    model = tf.keras.Model(inputs, outputs, name="YOLOv1_InceptionResNet")
    model.summary()
    return model



class YoloV1Loss(tf.keras.losses.Loss):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5, class_weights=None, name="yolo_v1_loss"):
        super().__init__(name=name)
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        object_mask = y_true[..., 4:5]  # (batch, S, S, 1)
        noobj_mask = 1.0 - object_mask

        pred_boxes = tf.reshape(y_pred[..., :B*5], (-1, S, S, B, 5))  # (batch, S, S, B, 5)
        true_box = tf.expand_dims(y_true[..., 0:4], axis=-2)  # (batch, S, S, 1, 4)

        ious = compute_iou(pred_boxes[..., 0:4], true_box)  # (batch, S, S, B)
        best_box = tf.argmax(ious, axis=-1)  # (batch, S, S)
        best_box = tf.one_hot(best_box, depth=B)  # (batch, S, S, B)
        best_box = tf.expand_dims(best_box, axis=-1)  # (batch, S, S, B, 1)

        object_mask_exp = tf.expand_dims(object_mask, axis=-2)  # (batch, S, S, 1, 1)
        responsible_mask = object_mask_exp * best_box  # (batch, S, S, B, 1)

        pred_xy = pred_boxes[..., 0:2]
        true_xy = tf.tile(true_box[..., 0:2], [1, 1, 1, B, 1])
        coord_xy_loss = tf.reduce_mean(responsible_mask * tf.square(pred_xy - true_xy))

        pred_wh = tf.maximum(pred_boxes[..., 2:4], 1e-6)
        true_wh = tf.maximum(tf.tile(true_box[..., 2:4], [1, 1, 1, B, 1]), 1e-6)
        coord_wh_loss = tf.reduce_mean(responsible_mask * tf.square(tf.sqrt(pred_wh) - tf.sqrt(true_wh)))

        coord_loss = self.lambda_coord * (coord_xy_loss + coord_wh_loss)

        pred_conf = pred_boxes[..., 4:5]
        true_conf = tf.tile(object_mask_exp, [1, 1, 1, B, 1])
        obj_loss = tf.reduce_mean(responsible_mask * tf.square(pred_conf - true_conf))
        noobj_loss = self.lambda_noobj * tf.reduce_mean((1 - responsible_mask) * tf.square(pred_conf))

        pred_class = y_pred[..., B*5:]
        true_class = y_true[..., B*5:]

        ce = tf.keras.losses.categorical_crossentropy(true_class, pred_class, from_logits=True)
        if self.class_weights is not None:
            class_indices = tf.argmax(true_class, axis=-1)
            weights = tf.gather(self.class_weights, class_indices)
            ce *= weights
        class_loss = tf.reduce_mean(tf.squeeze(object_mask, axis=-1) * ce)

        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        return tf.where(tf.math.is_nan(total_loss), tf.zeros_like(total_loss), total_loss)


if __name__ == "__main__":
    print("Building and saving classifier backbone...")
    classifier_model = build_classifier()
    classifier_model.save_weights("checkpoints/classifier_backbone.weights.h5")

    print("Building YOLOv1 model using pretrained backbone...")
    yolo_model = build_yolo_v1(
        backbone_weights="checkpoints/classifier_backbone.weights.h5", freeze_backbone=True)
    yolo_model.summary()
    tf.keras.utils.plot_model(classifier_model, to_file="classifier_model.png", show_shapes=True, show_layer_names=True)
    tf.keras.utils.plot_model(yolo_model,expand_nested=True, to_file="yolo_v1_model.png", show_shapes=True, show_layer_names=True)
