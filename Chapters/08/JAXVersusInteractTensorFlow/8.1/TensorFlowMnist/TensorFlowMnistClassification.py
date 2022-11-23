import numpy as np
import tensorflow as tf
import time

def setup():

    # The dataset will be downloaded from repository, make sure internet connection is active
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Fix the dimension
    train_images = np.expand_dims(train_images, axis = 3)

    # Convert to the tags of one-hot
    train_labels = tf.one_hot(train_labels, depth = 10)
    test_labels = tf.one_hot(test_labels, depth = 10)

    # Standardize the dataset with TensorFlow accepatable format
    trains = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1024).batch(256)
    tests = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(1024).batch(256)

    return trains, tests

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

def train(trains, tests, accelerator = "/CPU:0"):

    # Indicate that the GPU is used to compute the model while running TensorFlow
    with tf.device(accelerator):

        image = tf.keras.Input(shape = (28, 28, 1))
        logits = MnistModel()(image)

        model = tf.keras.Model(image, logits)

        for i in range(4):

            start = time.time()

            model.compile(optimizer = tf.keras.optimizers.SGD(1e-3), loss = tf.keras.losses.categorical_crossentropy, metrics = ["accuracy"])
            model.fit(trains, epochs = 2, validation_data = tests, verbose = 0)

            end = time.time()

            loss, accuracy = model.evaluate(tests)

            print("Test Loss", loss)
            print("Accuracy", accuracy)

            print(f"Test {i + 1} started, %.12fs is consumed" % (end - start))


def test():

    trains, tests = setup()

    train(trains, tests, accelerator = "/CPU:0")

if __name__ == '__main__':

    test()
