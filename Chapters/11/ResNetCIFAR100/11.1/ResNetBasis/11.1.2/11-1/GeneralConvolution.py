# Conv Implementation
import functools

import jax.random


def GeneralConvolution(dimension_numbers: int, out_channels: int, filter_shape, strides = None, padding = "VALID", weights_init= None, biases_init = jax.random.normal(1e-6)):

    """

    ...

    """

    def init_function():

        return None

    def apply_function():

        return None

    return init_function, apply_function

def Conv():

    # Wrap the main function
    conv = functools.partial(GeneralConvolution, ("NHWC", "HWIO", "NHWC"))

    return conv



