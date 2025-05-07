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

def shufflenet_v1_model(input_shape=(224, 224, 3), num_classes=1000, groups=3):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x:x/255.0)(inputs)
    x = tf.keras.layers.Conv2D(24, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    
    def shufflenet_unit(x, out_channels, groups):
        bottleneck_channels = out_channels // 4
        residual = GroupConv2D(bottleneck_channels, (1, 1), groups=groups)(x)
        residual = ChannelShuffle(groups)(residual)
        residual = tf.keras.layers.DepthwiseConv2D((3, 3), strides=1, padding='same', activation=None, use_bias=False)(residual)
        residual = GroupConv2D(out_channels, (1, 1), groups=groups)(residual)
        
        # Ensure x has the same shape as residual before addition
        projection = tf.keras.layers.Conv2D(out_channels, (1, 1), padding='same', use_bias=False)(x)
        
        return tf.keras.layers.ReLU()(projection + residual)
    
    for _ in range(4):
        x = shufflenet_unit(x, 144, groups)
    for _ in range(8):
        x = shufflenet_unit(x, 288, groups)
    for _ in range(4):
        x = shufflenet_unit(x, 576, groups)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

# Example usage
if __name__ == '__main__':
    model = shufflenet_v1_model(input_shape=(224, 224, 3), num_classes=1000, groups=3)
    model.summary()
