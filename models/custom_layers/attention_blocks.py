# Francesco Campigotto - 2156375

import tensorflow as tf
from keras import layers

class SEBlock(layers.Layer):
    def __init__(self, channels, reduction_ratio=16, trainable=True, dtype=None, **kwargs):
        super(SEBlock, self).__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.squeeze = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        se = self.squeeze(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = tf.reshape(se, [-1, 1, 1, tf.shape(inputs)[-1]])  
        return inputs * se


class CAMBlock(layers.Layer):
    def __init__(self, channels, reduction_ratio=16, trainable=True, dtype=None, **kwargs):
        super(CAMBlock, self).__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.channels = channels
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.global_max_pool = layers.GlobalMaxPooling2D()
        self.dense1 = layers.Dense(channels // reduction_ratio, activation='relu')
        self.dense2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        avg_pool = self.global_avg_pool(inputs)
        max_pool = self.global_max_pool(inputs)
        avg_out = self.dense2(self.dense1(avg_pool))
        max_out = self.dense2(self.dense1(max_pool))
        out = avg_out + max_out
        out = tf.reshape(out, [-1, 1, 1, self.channels])
        return inputs * out