import functools

import jax


def GeneralConvolutionTranspose(dimension_numbers, out_channels, filter_shape, strides = None, padding = "VALID", weights_init = None, biases_init = jax.nn.initializers.normal(1e-6)):

    lhs_spec, rhs_spec, out_spec = dimension_numbers

    conv1_transpose = functools.partial(GeneralConvolutionTranspose, ("NHC", "HIO", "NHC"))
    conv_transpose = functools.partial(GeneralConvolutionTranspose, ("NHWC", "HWIO", "NHWC"))
    

