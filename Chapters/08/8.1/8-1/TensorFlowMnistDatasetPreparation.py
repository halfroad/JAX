import tensorflow as tf
import numpy as np

def setup():

    # Download the dataset from network
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = np.expand_dims(train_images, axis = 3)
    train_labels = tf.one_hot(train_labels, depth = 10)
    test_labels = tf.one_hot(test_labels, depth = 10)

    trains = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1024).batch(256)
    tests = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(1024).batch(256)

    return trains, tests

def start():

    dataset = setup()

    print(dataset)

def main():

    start()

if __name__ == "__main__":

    main()
