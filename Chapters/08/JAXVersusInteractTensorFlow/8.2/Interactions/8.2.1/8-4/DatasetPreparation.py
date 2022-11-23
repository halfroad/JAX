import jax
import tensorflow_datasets as tfds

def setup():

    # Download the mnist dataset from tensorflow-dataset directly
    trains, tests = tfds.load(name = "mnist", split = [tfds.Split.TRAIN, tfds.Split.TRAIN], batch_size = -1, data_dir = "../../../../../../../Shares/Datasets/MNIST/")

    tests = tfds.as_numpy(tests)
    trains = tfds.as_numpy(trains)

    train_images, train_labels = trains["image"], trains["label"]
    test_images, test_labels = tests["image"], tests["label"]

    _, height, width, channels = train_images.shape

    # Get the dimension of dataset
    pixels = height * width * channels

    # Set the number of classifications
    output_dimensions = 10

    # Adapt the inputs dimension
    train_images = jax.numpy.reshape(train_images, (-1, pixels))
    test_images = jax.numpy.reshape(test_images, (-1, pixels))

    # One Hot the labels
    train_labels = one_hot(inputs = train_labels, k = output_dimensions)
    test_labels = one_hot(inputs = train_labels, k = output_dimensions)

    return (train_images, train_labels), (test_images, test_labels)

def one_hot(inputs, k = 10, dtype = jax.numpy.float32):
    
    matches = jax.numpy.array(inputs[:, None] == jax.numpy.arange(k), dtype)

    return matches
