import tensorflow as tf

def Lenet5(input_shape = [84,84,4], num_classes=4):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='selu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='selu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='selu')(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def Lenet5_advantage(input_shape = [84,84,4], num_classes=4):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='selu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='selu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='selu')(x)
    value = tf.keras.layers.Dense(1)(x)
    advantage = tf.keras.layers.Dense(num_classes)(x)
    outputs = tf.keras.layers.Add()([value, advantage])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

    

if __name__ == "__main__":
    model = Lenet5()
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)