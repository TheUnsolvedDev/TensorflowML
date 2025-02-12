import silence_tensorflow.auto
import tensorflow as tf



def fire_module(inputs, s1x1, e1x1, e3x3, name="fire"):
    squeeze = tf.keras.layers.Conv2D(s1x1, (1,1), strides=1, padding="valid", activation="relu", name=f"{name}_squeeze")(inputs)
    
    expand_1x1 = tf.keras.layers.Conv2D(e1x1, (1,1), strides=1, padding="valid", activation="relu", name=f"{name}_expand1x1")(squeeze)
    expand_3x3 = tf.keras.layers.Conv2D(e3x3, (3,3), strides=1, padding="same", activation="relu", name=f"{name}_expand3x3")(squeeze)
    
    return tf.keras.layers.Concatenate(name=f"{name}_concat")([expand_1x1, expand_3x3])

def squeezenet_complexbypass_model(input_shape=(224,224,3), num_classes=100):
    """ SqueezeNet v0 with complex bypass (skip) connections """

    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    
    conv1 = tf.keras.layers.Conv2D(96, (7,7), strides=2, padding="same", activation="relu", name="conv1")(inputs)
    pool1 = tf.keras.layers.MaxPooling2D((3,3), strides=2, name="pool1")(conv1)
    
    fire2 = fire_module(pool1, 16, 64, 64, name="fire2")
    fire3 = fire_module(fire2, 16, 64, 64, name="fire3")
    bypass_23 = tf.keras.layers.Add(name="bypass_23")([fire2, fire3])  
    
    fire4 = fire_module(bypass_23, 32, 128, 128, name="fire4")
    pool2 = tf.keras.layers.MaxPooling2D((3,3), strides=2, name="pool2")(fire4)
    
    fire5 = fire_module(pool2, 32, 128, 128, name="fire5")
    bypass_45 = tf.keras.layers.Add(name="bypass_45")([pool2, fire5])  
    
    fire6 = fire_module(bypass_45, 48, 192, 192, name="fire6")
    fire7 = fire_module(fire6, 48, 192, 192, name="fire7")
    fire8 = fire_module(fire7, 64, 256, 256, name="fire8")
    
    pool3 = tf.keras.layers.MaxPooling2D((3,3), strides=2, name="pool3")(fire8)
    
    fire9 = fire_module(pool3, 64, 256, 256, name="fire9")
    bypass_89 = tf.keras.layers.Add(name="bypass_89")([pool3, fire9])  
    
    drop = tf.keras.layers.Dropout(0.5, name="dropout")(bypass_89)
    conv10 = tf.keras.layers.Conv2D(num_classes, (1,1), strides=1, padding="same", activation="relu", name="conv10")(drop)
    
    avg_pool = tf.keras.layers.AveragePooling2D((13,13), strides=1, name="global_avg_pool")(conv10)
    flatten = tf.keras.layers.Flatten(name="flatten")(avg_pool)
    outputs = tf.keras.layers.Softmax(name="softmax")(flatten)
    
    model = tf.keras.Model(inputs, outputs, name="SqueezeNetV0_Res")
    return model


if __name__ == '__main__':
    model1 = squeezenet_complexbypass_model()
    model1.summary(expand_nested=True)
