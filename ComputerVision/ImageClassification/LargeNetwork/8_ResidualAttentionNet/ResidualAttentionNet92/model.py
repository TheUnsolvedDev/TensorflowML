import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *


def Residual_Unit(input, in_channel, out_channel, stride=1):
    shortcut = input
    shortcut = tf.keras.layers.Conv2D(out_channel, (1, 1), padding='same', strides=stride)(shortcut)
    x = tf.keras.layers.BatchNormalization()(input)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(in_channel, (1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(in_channel, (3, 3), padding='same', strides=stride)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(out_channel, (1, 1), padding='same')(x)
    x = tf.keras.layers.Add()([x, shortcut])
    return x

def Attention_Block(input, skip):
    p, t, r = 1, 2, 1
    skip_connections = []
    in_channel = input.shape[-1]
    out_channel = in_channel
    for _ in range(p):
        x = Residual_Unit(input, in_channel, out_channel)
    for _ in range(t):
        Trunck_output = Residual_Unit(x, in_channel, out_channel)
    x = tf.keras.layers.MaxPooling2D(padding='same')(x)
    for _ in range(r):
        x = Residual_Unit(x, in_channel, out_channel)
    if x.shape[1] % 4 == 0:
        for i in range(skip - 1):
            skip_connections.append(Residual_Unit(x, in_channel, out_channel))
            x = tf.keras.layers.MaxPooling2D(padding='same')(x)
            for _ in range(r):
                x = Residual_Unit(x, in_channel, out_channel)
        skip_connections = list(reversed(skip_connections))
        for i in range(skip - 1):
            for _ in range(r):
                x = Residual_Unit(x, in_channel, out_channel)
            x = tf.keras.layers.UpSampling2D()(x)
            x = tf.keras.layers.Add()([x, skip_connections[i]])
    for i in range(r):
        x = Residual_Unit(x, in_channel, out_channel)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(out_channel, (1, 1))(x)
    x = tf.keras.layers.Conv2D(out_channel, (1, 1))(x)
    soft_mask_output = tf.keras.layers.Activation('sigmoid')(x)
    soft_mask_output = tf.keras.layers.Lambda(lambda x: x + 1)(soft_mask_output)
    output = tf.keras.layers.Multiply()([soft_mask_output, Trunck_output])
    for i in range(p):
        output = Residual_Unit(output, in_channel, out_channel)
    return output

def residual_attentionnet92_model(input_shape=[32, 32, 3], num_classes=10, dropout=None, regularization=0.01):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)

    in_channel = 32
    out_channel = in_channel * 4
    x = Residual_Unit(x, in_channel, out_channel)
    x = Attention_Block(x, skip=2)

    in_channel = out_channel // 2
    out_channel = in_channel * 4
    x = Residual_Unit(x, in_channel, out_channel, stride=2)
    x = Attention_Block(x, skip=1)
    x = Attention_Block(x, skip=1)

    in_channel = out_channel // 2
    out_channel = in_channel * 4
    x = Residual_Unit(x, in_channel, out_channel, stride=2)
    x = Attention_Block(x, skip=1)
    x = Attention_Block(x, skip=1)
    x = Attention_Block(x, skip=1)

    in_channel = out_channel // 2
    out_channel = in_channel * 4
    x = Residual_Unit(x, in_channel, out_channel, stride=1)
    x = Residual_Unit(x, in_channel, out_channel)
    x = Residual_Unit(x, in_channel, out_channel)

    x = tf.keras.layers.AveragePooling2D(pool_size=4, strides=1)(x)
    x = tf.keras.layers.Flatten()(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(regularization), activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    models = [
        residual_attentionnet92_model,]
    
    for model_fn in models:
        model = model_fn()
        model.summary()