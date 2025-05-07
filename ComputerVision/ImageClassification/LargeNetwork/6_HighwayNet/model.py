import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from config import *



class HighwayBlock(tf.keras.layers.Layer):
    def __init__(self, units, t_bias=-2.0, acti_h=tf.nn.relu, acti_t=tf.nn.sigmoid):
        super(HighwayBlock, self).__init__()
        self.units = units
        self.t_bias = t_bias
        self.acti_h = acti_h
        self.acti_t = acti_t

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(name="W", shape=(input_dim, self.units), initializer="random_normal")
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="random_normal")
        self.W_T = self.add_weight(name="W_T", shape=(input_dim, self.units), initializer="random_normal")
        self.b_T = self.add_weight(name="b_T", shape=(self.units,), initializer=tf.constant_initializer(self.t_bias))

    def call(self, inputs):
        h = self.acti_h(tf.matmul(inputs, self.W) + self.b)
        t = self.acti_t(tf.matmul(inputs, self.W_T) + self.b_T)
        return h * t + inputs * (1.0 - t)

def highway_net_model(input_shape = [INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10, num_layers=3, t_bias=-2.0):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

    x = tf.keras.layers.Flatten()(x) 
    x = tf.keras.layers.Dense(50)(x)  

    for _ in range(num_layers):
        x = HighwayBlock(50, t_bias=t_bias)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)



if __name__ == "__main__":
    model = highway_net_model()
    model.summary(expand_nested=True)
    
    tf.keras.utils.plot_model(
        model, to_file=highway_net_model.__name__+'.png', show_shapes=True)