import array
import gzip
import os
import ssl
import struct
import urllib.request
import jax.numpy

from os import path
from tqdm import tqdm

_MNIST = "../../../../../Exclusion/Datasets/MNIST"

def _download(url, name):

    """

    Download an url to a file in JAX data temporary directory

    """

    if not path.exists(_MNIST):

        os.makedirs(_MNIST)

    out_file = path.join(_MNIST, name)

    if not path.isfile(out_file):

        ssl._create_default_https_context = ssl._create_unverified_context

        with tqdm(unit = "B", unit_scale = True, unit_divisor = 1024, miniters = 1, desc = name) as bar:

            urllib.request.urlretrieve(url, out_file, reporthook = report_hook(bar))

        print(f"Downloaded {url} to {_MNIST}")

def report_hook(bar: tqdm):

    """

    Progress Bar of tqdm for downloads

    """

    def hook(block_counter = 0, block_size = 1, total_size = None):

        if total_size is not None:

            bar.total = total_size

        bar.update(block_counter * block_size - bar.n)

    return hook

def _partial_flatten(inputs):

    """

    Flatten all but the first dimension of an array

    """

    return jax.lax.expand_dims(inputs, [-1]) / jax.numpy.float32(255.)

def _one_hot_nojit(inputs, k = 10, dtype = jax.numpy.float32):

    """

    Create a one-hot encoding of inputs of size k.

    """

    return jax.numpy.array(inputs[:, None] == jax.numpy.arange(k), dtype)

def mnist_raw():

    """

    Download and parse the raw MNIST dataset.

    """

    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(file):

        with gzip.open(file, "rb") as handler:

            _ = struct.unpack(">II", handler.read(8))

            return jax.numpy.array(array.array("B", handler.read()), dtype = jax.numpy.uint8)

    def parse_images(file):

        with gzip.open(file, "rb") as handler:

            _, number, rows, columns = struct.unpack(">IIII", handler.read(16))

            return jax.numpy.array(array.array("B", handler.read()), dtype = jax.numpy.uint8).reshape(number, rows, columns)

    for name in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:

        url = path.join(_MNIST, name)

        if not path.exists(url):
            _download(base_url + name, name)

    train_images = parse_images(path.join(_MNIST, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_MNIST, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_MNIST, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_MNIST, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels

def mnist(permute_train = False):

    """

    Download, parse and process the MNIST data to unit scale and one-hot labels

    """

    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images = _partial_flatten(train_labels)
    test_images = _partial_flatten(test_images)
    train_labels = _one_hot_nojit(train_labels)
    test_labels = _one_hot_nojit(test_labels)

    if permute_train:

        permutation = jax.random.permutation(train_images.shape[0])

        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

    return train_images, train_labels, test_images, test_labels
