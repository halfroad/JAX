import numpy as np
import tensorflow as tf

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

