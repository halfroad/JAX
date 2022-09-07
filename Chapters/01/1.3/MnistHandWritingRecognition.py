import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp


def prepare():
    images_train = jnp.load("mnist_images_train.npy")
    labels_train = jnp.load("mnist_labels_train.npy")


def onehot(x, k=10, dtype=jnp.float32):

    """

    Create a one-hot encoding of x of k.

    """

    return jnp.array(x[:, None] == jnp.arange(k), dtype)
