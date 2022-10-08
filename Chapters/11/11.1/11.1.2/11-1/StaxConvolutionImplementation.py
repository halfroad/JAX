import functools

import jax

# Convolution general function
def GeneralConvolute(dimension_numbers, out_channels, filter_shape, strides = None, padding = "VALID", weight_init = None, bias_init = jax.nn.initializers.normal(1e-6)):

    # return init_function, apply_function
    print("general_convolution")

def Convolute():

    """

    There are 2 steps to create a convolution:
    1. General function of convolution.
    2. Wrap the main function with formatation

    Wrap the main funciton

    """

    convolute = functools.partial(GeneralConvolute, ("NHWC", "HWIO", "NHWC"))

    return convolute

def BatchNormalize(axis = (0, 1, 2), epsilon = 1e-5, center = True, scale = True, beta_init = jax.nn.initializers.zeros, gamma_init = jax.nn.initializers.ones):

    # return init_function, apply_init
    print("batch_normalize")

def Dense(out_dimenstion = 10, weight_init = jax.nn.initializers.glorot_normal(), bias_init = jax.nn.initializers.normal()):

    # return init_function, apply_function
    print("Dense")

def Pool():

    # Size of pooling window
    window_shape = (3, 3)

    # Size of step
    strides = (2, 2)

    jax.example_libraries.stax.AvgPool(window_shape = window_shape, stides = strides)
    # jax.example_libraries.stax.AvgPool(window_shape = window_shape, stides = strides, padding = "VALID", spce = None)

    # return init_function, apply_function

def train():

    # Number of kernels of convolution, this number yields the dimension after the data processed
    filter_number = 64

    # Size of convolution kernel
    filter_size = (3, 3)

    # Length of step
    strides = (2, 2)

    Convolute(filter_number, filter_size, strides)
    Convolute(filter_number, filter_size, strides, padding = "SAME")
