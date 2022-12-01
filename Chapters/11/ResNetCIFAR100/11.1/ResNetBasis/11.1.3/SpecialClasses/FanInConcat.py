import jax.numpy


def fanInConcat(axis = -1):

    """
    Layer construction function for a fan-in concatenation layer.
    """

    def init_function(rng, input_shape):

        ax = axis % len(input_shape[0])
        concat_size = sum(shape[ax] for shape in input_shape)
        out_shape = input_shape[0][: ax] + (concat_size,) + input_shape[0][ax + 1]

        return out_shape, ()

    def apply_function(params, inputs, **kwargs):

        return jax.numpy.concatenate(inputs, axis)

    return init_function, apply_function

