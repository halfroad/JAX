import jax

def partial_flatten(inputs):

    """

    Flatten all but the first dimension of an array

    jax.lax.expand_dims(inputs, [-1]): [60000, 28, 28] -> [60000, 28, 28, 1]
    jax.lax.expand_dims(inputs, [1, 2]): [60000, 28, 28] -> [60000, 1, 1, 28, 28]

    """
    inputs = jax.lax.expand_dims(inputs, [-1])  # [60000, 28, 28] -> [60000, 28, 28, 1]

    return inputs / 255.

def one_hot_nojit(inputs, k = 10, dtype = jax.numpy.float32):

    """

    Create a one-hot encoding of inputs of size k.

    """

    return jax.numpy.array(inputs[:, None] == jax.numpy.arange(k), dtype)
