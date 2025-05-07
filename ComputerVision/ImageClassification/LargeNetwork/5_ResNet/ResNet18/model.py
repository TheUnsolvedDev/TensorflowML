import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *

kaiming_normal = tf.keras.initializers.VarianceScaling(scale=2.0,mode= 'fan_out',distribution = 'untruncated_normal')

def conv3x3(x,out_planes,stride=1,name=None):
    x = tf.keras.layers.ZeroPadding2D(padding=1,name=f'{name}_pad')(x)
    x = tf.keras.layers.Conv2D(out_planes,kernel_size=3,strides=stride,kernel_initializer=kaiming_normal,name=f'{name}_conv')(x)
    return x

def basic_block(x, planes, stride= 1,downsample=None,name=None):
    identity = x
    
    out = conv3x3(x,planes,stride,name=f'{name}_conv1')
    out = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,name=f'{name}_bn1')(out)
    out = tf.keras.layers.ReLU(name=f'{name}_relu1')(out)
    
    out = conv3x3(out,planes,name=f'{name}_conv2')
    out = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,name=f'{name}_bn2')(out)
    
    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)
            
    out = tf.keras.layers.Add()([out,identity])
    out = tf.keras.layers.ReLU(name=f'{name}_relu2')(out)
    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            tf.keras.layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet18_model(input_shape = (INPUT_SIZE[0],INPUT_SIZE[1],3),num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x:x/255.0)(inputs)
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, kernel_initializer=kaiming_normal)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = make_layer(x, 64, 2, stride=1, name='layer1')
    x = make_layer(x, 128, 2, stride=2, name='layer2')
    x = make_layer(x, 256, 2, stride=2, name='layer3')
    x = make_layer(x, 512, 2, stride=2, name='layer4')
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    initializer = tf.keras.initializers.RandomUniform(-1.0/tf.sqrt(512.0), 1.0/tf.sqrt(512.0))
    outputs = tf.keras.layers.Dense(num_classes, kernel_initializer=initializer,name='fc')(x)
    return tf.keras.Model(inputs = inputs,outputs = outputs)


if __name__ == "__main__":
    model = resnet18_model()
    model.summary(expand_nested=True)
    pass