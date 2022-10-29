import jax.random

"""

Paragraph 2.1.2, Implementation of Fully-Connected Layer
Page 18

"""

def Dense(shape = [2, 1]):

    # 0 is the random send, can be any number
    key = jax.random.PRNGKey(0)

    weights = jax.random.normal(key, shape = shape)
    biases = jax.random.normal(key, shape = (shape[-1],))

    parameters = [weights, biases]

    # apply_fun is an internal feature of python, which is intrinsic fucntion
    def apply_func(inputs):

        weight, bias = parameters

        return jax.numpy.dot(inputs, weight) + bias

    return apply_func




