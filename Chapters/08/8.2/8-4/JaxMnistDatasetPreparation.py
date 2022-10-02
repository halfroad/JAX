import jax
import tensorflow_datasets as tfds

# One-Hot encoding the labels
def one_hot(inputs, k = 10, dtype = jax.numpy.float32):

    """

    Create a one-hot encoding of inputs of size k.

    """

    return jax.numpy.array(inputs[:, None] == jax.numpy.arange(k), dtype)

def setup():

    # Load the datasets from tensorflow-datasets
    trains = tfds.load("mnist", split = tfds.Split.TRAIN, batch_size = -1)
    trains = tfds.as_numpy(trains)

    train_images, train_labels, = trains["image"], trains["label"]

    _, height, width, channels = train_images.shape

    # Read the dimensions
    number_pixels = height * width * channels

    # Set the number of classifications
    output_dimensions = 10

    # Change the dimensions of input dataset
    train_images = jax.numpy.reshape(train_images, (-1, number_pixels))

    # One-Hot
    train_labels = one_hot(train_labels, k = output_dimensions)

    tests = tfds.load("mnist", split = tfds.Split.TEST, batch_size = -1)
    tests = tfds.as_numpy(tests)

    test_images, test_labels = tests["image"], tests["label"]

    test_images = jax.numpy.reshape(test_images, (-1, number_pixels))
    test_labels = one_hot(test_labels, k = output_dimensions)

    return (train_images, train_labels), (test_images, test_labels), number_pixels, output_dimensions

