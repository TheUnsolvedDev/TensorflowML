import tensorflow as tf

class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, groups=1):
        super(GroupConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.groups = groups
        self.convs = []

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.group_filters = self.filters // self.groups
        for _ in range(self.groups):
            self.convs.append(tf.keras.layers.Conv2D(self.group_filters, self.kernel_size, strides=self.strides, padding='same', activation=None, use_bias=False))

    def call(self, x):
        group_list = tf.split(x, num_or_size_splits=self.groups, axis=-1)
        group_convs = [conv(group) for conv, group in zip(self.convs, group_list)]
        return tf.keras.layers.Concatenate()(group_convs)

class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, num_groups):
        super(ChannelShuffle, self).__init__()
        self.num_groups = num_groups

    def call(self, x):
        if self.num_groups == 1:
            return x
        batch_size = tf.shape(x)[0]
        height, width, channels = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        group_size = channels // self.num_groups
        x = tf.reshape(x, [batch_size, height, width, self.num_groups, group_size])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [batch_size, height, width, channels])
        return x


class ShuffleNetV2Unit(tf.keras.layers.Layer):
    def __init__(self, out_channels, stride):
        super(ShuffleNetV2Unit, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.mid_channels = max(2, out_channels // 2)  # Ensure mid_channels is even
        self.ensure_even_channels = tf.keras.layers.Conv2D(self.out_channels + (self.out_channels % 2), (1, 1), padding='same', use_bias=False)

    def call(self, x):
        x = self.ensure_even_channels(x)  # Ensure even number of channels
        if self.stride == 1:
            left, right = tf.split(x, num_or_size_splits=2, axis=-1)
            right = tf.keras.layers.Conv2D(self.mid_channels, (1, 1), padding='same', use_bias=False)(right)
            right = tf.keras.layers.BatchNormalization()(right)
            right = tf.keras.layers.ReLU()(right)
            right = tf.keras.layers.DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False)(right)
            right = tf.keras.layers.BatchNormalization()(right)
            right = tf.keras.layers.Conv2D(self.mid_channels, (1, 1), padding='same', use_bias=False)(right)
            right = tf.keras.layers.BatchNormalization()(right)
            x = tf.keras.layers.Concatenate()([left, right])
        else:
            left = tf.keras.layers.DepthwiseConv2D((3, 3), strides=self.stride, padding='same', use_bias=False)(x)
            left = tf.keras.layers.BatchNormalization()(left)
            left = tf.keras.layers.Conv2D(self.mid_channels, (1, 1), padding='same', use_bias=False)(left)
            left = tf.keras.layers.BatchNormalization()(left)
            right = tf.keras.layers.Conv2D(self.mid_channels, (1, 1), padding='same', use_bias=False)(x)
            right = tf.keras.layers.BatchNormalization()(right)
            right = tf.keras.layers.ReLU()(right)
            right = tf.keras.layers.DepthwiseConv2D((3, 3), strides=self.stride, padding='same', use_bias=False)(right)
            right = tf.keras.layers.BatchNormalization()(right)
            right = tf.keras.layers.Conv2D(self.mid_channels, (1, 1), padding='same', use_bias=False)(right)
            right = tf.keras.layers.BatchNormalization()(right)
            x = tf.keras.layers.Concatenate()([left, right])
        return ChannelShuffle(2)(x)

def shufflenet_v2_model(input_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    
    for _ in range(4):
        x = ShuffleNetV2Unit(116, 1)(x)
    for _ in range(8):
        x = ShuffleNetV2Unit(232, 2)(x)
    for _ in range(4):
        x = ShuffleNetV2Unit(464, 2)(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

if __name__ == "__main__":
    # Example usage
    model = shufflenet_v2_model(input_shape=(224, 224, 3), num_classes=1000)
    model.summary()
