import jax.random


def setup():

    key = jax.random.PRNGKey(15)
    input_shape = (1000,)

    inputs = jax.random.normal(key, shape = input_shape)

    weight = 0.929
    bias = 0.214

    # Initial params
    params = jax.random.normal(key, (2, ))

    genuines = inputs * weight + bias

    return (key, input_shape), (weight, bias), params, (inputs, genuines)
