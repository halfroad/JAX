import tensorflow as tf

class MnistModel(tf.keras.layers.Layer):

    def __init__(self):

        super(MnistModel, self).__init__()

        self.bn1 = None
        self.conv1 = None

        self.conv2 = None
        self.bn2 = None
        self.dense = None

    def build(self, input_shape):

        self.conv1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, activation = tf.nn.relu)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, activation = tf.nn.relu)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.dense = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid)

        # Be sure to call this in the end
        super(MnistModel, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):

        embedding = inputs

        embedding = self.conv1(embedding)
        embedding = self.bn1(embedding)

        embedding = self.conv2(embedding)
        embedding = self.bn2(embedding)

        embedding = tf.keras.layers.Flatten()(embedding)

        logits = self.dense(embedding)

        return logits



