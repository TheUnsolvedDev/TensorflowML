import tensorflow as tf

class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, num_groups, **kwargs):
        super().__init__(**kwargs)
        self.num_groups = num_groups

    def call(self, inputs):
        batch_size, height, width, channels = tf.unstack(tf.shape(inputs))
        group_channels = channels // self.num_groups
        inputs = tf.reshape(inputs, [batch_size, height, width, self.num_groups, group_channels])
        inputs = tf.transpose(inputs, [0, 1, 2, 4, 3])
        return tf.reshape(inputs, [batch_size, height, width, channels])

class ShuffleUnit(tf.keras.layers.Layer):
    def __init__(self, out_channels, stride, groups, left_ratio, spatial_down, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.left_ratio = left_ratio
        self.spatial_down = spatial_down
        self.depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=self.stride, padding='same')
        self.channel_shuffle = ChannelShuffle(self.groups)

    def build(self, input_shape):
        depth_in = input_shape[-1]
        depth_left = int(depth_in * self.left_ratio)
        depth_right = depth_in - depth_left

        if self.spatial_down:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=self.stride, padding='same'),
                tf.keras.layers.Conv2D(self.out_channels, kernel_size=1, padding='same', activation=None)
            ])
        else:
            self.shortcut = lambda x: x[:, :, :, :depth_left]

        self.residual_1x1 = tf.keras.layers.Conv2D(depth_right, kernel_size=1, padding='same', activation=None)
        self.residual_depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=self.stride, padding='same')
        self.residual_1x1_out = tf.keras.layers.Conv2D(depth_right, kernel_size=1, padding='same', activation=None)

    def call(self, inputs):
        shortcut = self.shortcut(inputs)
        residual = self.residual_1x1(inputs)
        residual = self.residual_depthwise(residual)
        residual = self.residual_1x1_out(residual)
        output = tf.concat([shortcut, residual], axis=-1)
        return self.channel_shuffle(output)


def shufflenet_v2_model(input_shape, num_classes, depth_multiplier=1.0):
    depths_dict = {0.5: (48, 96, 192, 1024),
                   1.0: (116, 232, 464, 1024),
                   1.5: (176, 352, 704, 1024),
                   2.0: (244, 488, 976, 2048)}
    depths = depths_dict[depth_multiplier]
    
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)
    x = tf.keras.layers.Conv2D(24, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    
    for depth in depths[:-1]:
        x = ShuffleUnit(depth, 2, 2, 0.5, True)(x)
        for _ in range(3):
            x = ShuffleUnit(depth, 1, 2, 0.5, False)(x)
    
    x = tf.keras.layers.Conv2D(depths[-1], 1, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if num_classes:
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, x)

if __name__ == "__main__":
    model = shufflenet_v2_model((224, 224, 3), 1000)
    model.summary()