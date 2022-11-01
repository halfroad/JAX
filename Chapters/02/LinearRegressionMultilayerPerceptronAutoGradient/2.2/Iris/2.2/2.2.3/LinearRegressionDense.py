import jax.random

"""

Paragraph 2.2.3, Model Design based on JAX Linear Regression
Page 25
First Step: Fully-Connected Layer

"""

def Dense(shape = [4, 1]):

    def init_function(input_shape = shape):

        prng = jax.random.PRNGKey(15)

        weights, biases = jax.random.normal(prng, shape = input_shape), jax.random.normal(prng, shape = (shape[-1]))

        return weights, biases

    def apply_function(inputs, parameters):

        weight, bias = parameters

        return jax.numpy.dot(inputs, weight) + bias

    return init_function, apply_function

"""

Paragraph 2.2.3, Model Design based on JAX Linear Regression
Page 25
Second Step: Loss Function

"""
def linear_loss_function(parameters, inputs, genuines, apply_function):

    """

    Loss Function: MSE (Mean Squared Error)
    x = inputs, y = genuines

    g(x) = (f(x) - y)²

    Ordinary Least Squares

    """

    predictions = apply_function(inputs, parameters)
    mse = jax.numpy.power(predictions - genuines, 2.0)

    return mse
