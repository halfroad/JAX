from jax.example_libraries import stax
from jax.example_libraries.stax import Conv, Relu, Flatten, Dense, LogSoftmax


def CharactersConvolutionalNeuralNetworks(number_classes):

    return stax.serial(

        Conv(1, (3, 3)),
        Relu,

        Conv(1, (5, 5)),
        Relu,

        Conv(1, (3, 3)),
        Relu,

        Flatten,

        Dense(32),
        Relu,
        Dense(number_classes),
        LogSoftmax
    )
