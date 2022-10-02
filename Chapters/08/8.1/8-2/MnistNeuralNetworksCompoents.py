import tensorflow as tf

class MnistNeuralNetworksCompoentsLayers(tf.keras.layers.Layer):

    def __init__(self):

        super(MnistNeuralNetworksCompoentsLayers, self).__init__()

    def build(self, input_shape):

        self.conv1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, activation = tf.nn.relu)
        self.batchNormalization1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, activation = tf.nn.relu)
        self.batchNormalization2 = tf.keras.layers.BatchNormalization()

        self.dense = tf.keras.layers.Dense(10, activation = tf.nn.sigmoid)

        super(MnistNeuralNetworksCompoentsLayers, self).build(input_shape)

    def call(self, inputs):

        embedding = inputs

        embedding = self.conv1(embedding)
        embedding = self.batchNormalization1(embedding)

        embedding = self.conv2(embedding)
        embedding = self.batchNormalization2(embedding)

        embedding = tf.keras.layers.Flatten()(embedding)

        logits = self.dense(embedding)

        return logits
