import jax

from typing import NamedTuple


class Params(NamedTuple):

    # N-dimensional array
    weight: jax.numpy.ndarray
    bias: jax.numpy.ndarray

def setup():

    key = jax.random.PRNGKey(15)
    shape = (1, )

    weight = 1.86
    bias = 1.68

    params = init(key, shape)

    inputs = jax.random.normal(key, (1000,))
    genuines = inputs * weight + bias

    return (key, shape), (weight, bias, params), (inputs, genuines)

# The body of model
def model(params: Params, inputs):

    prediction = inputs * params.weight + params.bias

    return prediction

# The approach to initialize the params
def init(key, shape):

    weight = jax.random.normal(key = key, shape = shape)
    bias = jax.random.normal(key = key + 1, shape = shape)

    params = Params(weight = weight, bias = bias)

    return params

# Loss Function
def loss_function(params: Params, inputs, genuines):

    predictions = model(params = params, inputs = inputs)
    loss = (predictions - genuines) ** 2
    mse = jax.numpy.mean(loss)

    return mse

# Stochatic Gradient Descent Optimizer
def optimizer(params: Params, inputs, genuines, learn_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)

    params = jax.tree_util.tree_map(lambda param, gradient: param - learn_rate * gradient, params, gradients)

    return params

def train():

    (key, shape), (weight, bias, params), (inputs, genuines) = setup()

    print("Initial params = ", params)

    for i in range(6000):

        params = optimizer(params, inputs, genuines)

        print(f"weight = {params.weight}, bias = {params.bias}")

    print(f"weight = {params.weight}, bias = {params.bias}")

if __name__ == '__main__':

    train()


