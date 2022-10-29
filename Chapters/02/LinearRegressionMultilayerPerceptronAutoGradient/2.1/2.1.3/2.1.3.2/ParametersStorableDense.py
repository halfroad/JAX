import jax.random

"""

Paragraph 2.1.3, Parameters Storable Fully Connected Layer
Page 20

"""

def Dense(shape = [2, 1]):

    def init_function(input_shape = shape):

        key = jax.random.PRNGKey(15)

        weights, biases = jax.random.normal(key, shape = shape), jax.random.normal(key, shape = (shape[-1],))

        return weights, biases

    def apply_function(inputs, parameters):

        weight, bias = parameters

        return jax.numpy.dot(inputs, weight) + bias

    return init_function, apply_function

