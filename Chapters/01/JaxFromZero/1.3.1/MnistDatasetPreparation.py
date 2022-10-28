import jax.numpy
import tensorflow_datasets
from jax.example_libraries import stax


def parepare():

    (images_train, images_labels), (images_test, images_test) = tensorflow_datasets.load(name = "mnist", data_dir = "../../../../Shares/Datasets/MNIST/", split = [tensorflow_datasets.Split.TRAIN, tensorflow_datasets.Split.TEST])

    return (images_train, images_labels), (images_test, images_test)

def one_hot_no_jit(inputs, k = 10, dtype = jax.numpy.float32):

    """

    Create a one-hot encoding of inputs of size k.

    """

    return jax.numpy.array(inputs[:, None] == jax.numpy.arange(k), dtype = dtype)

def model():

    """

    {Dense(1024) -> ReLU} X 2 -> Dense(10) -> Lohsoftmax

    """

    init_random_parameters, predict = stax.serial(

        stax.Dense(1024),
        stax.Relu,

        stax.Dense(1024),
        stax.Relu,

        stax.Dense(10),
        stax.LogSoftmax
    )

    return init_random_parameters, predict

def start():

    (images_train, images_labels), (images_test, images_test) = parepare()

