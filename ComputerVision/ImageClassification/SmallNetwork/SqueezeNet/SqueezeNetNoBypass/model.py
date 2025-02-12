import silence_tensorflow.auto
import tensorflow as tf

def fire_module(inputs, s1x1, e1x1, e3x3, name="fire"):
    """Fire Module with squeeze and expand layers"""
    squeeze = tf.keras.layers.Conv2D(s1x1, (1, 1), padding="valid", activation="relu", name=f"{name}_squeeze")(inputs)
    expand_1x1 = tf.keras.layers.Conv2D(e1x1, (1, 1), padding="valid", activation="relu", name=f"{name}_expand_1x1")(squeeze)
    expand_3x3 = tf.keras.layers.Conv2D(e3x3, (3, 3), padding="same", activation="relu", name=f"{name}_expand_3x3")(squeeze)
    
    return tf.keras.layers.Concatenate(name=f"{name}_concat")([expand_1x1, expand_3x3])


def squeezenet_model(input_shape=(224, 224, 3), num_classes=1000):
    """SqueezeNet V0"""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)
    x = tf.keras.layers.Conv2D(96, (7, 7), strides=2, padding="same", activation="relu", name="conv1")(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same", name="pool1")(x)

    x = fire_module(x, 16, 64, 64, name="fire2")
    x = fire_module(x, 16, 64, 64, name="fire3")
    x = fire_module(x, 32, 128, 128, name="fire4")
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same", name="pool2")(x)

    x = fire_module(x, 32, 128, 128, name="fire5")
    x = fire_module(x, 48, 192, 192, name="fire6")
    x = fire_module(x, 48, 192, 192, name="fire7")
    x = fire_module(x, 64, 256, 256, name="fire8")
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same", name="pool3")(x)

    x = fire_module(x, 64, 256, 256, name="fire9")
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    x = tf.keras.layers.Conv2D(num_classes, (1, 1), padding="same", activation="relu", name="conv10")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = tf.keras.layers.Activation("softmax", name="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="SqueezeNet_V0")

if __name__ == '__main__':
    model1 = squeezenet_model()
    model1.summary(expand_nested=True)
