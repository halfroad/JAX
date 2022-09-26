import jax
import jax.numpy as jnp
import time

from typing import NamedTuple

# Note the generation of parameters
class Parameters(NamedTuple):

    weight: jnp.array
    bias: jnp.array

def setup():

    key = jax.random.PRNGKey(17)

    weight = .929
    bias = .214

    inputs = jax.random.normal(key, shape = (1000, 1))
    genuines = weight * inputs + bias

    return key, weight, bias, inputs, genuines

def init(key):

    weight = jax.random.normal(key, (1,))
    bias = jax.random.normal(key + 1, (1,))

    return Parameters(weight, bias)

# Model
@jax.jit
def model(parameters: Parameters, inputs):

    predictions = parameters.weight * inputs + parameters.bias

    return predictions

@jax.jit
def loss_function(parameters: Parameters, inputs, genuines):

    predictions = model(parameters, inputs)
    losses = (predictions - genuines) ** 2

    return jnp.mean(losses)

@jax.jit
def optimizer(parameters: Parameters, inputs, geniunes, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    derivatives = grad_loss_function(parameters, inputs, geniunes)

    parameters = jax.tree_util.tree_map(lambda parameter, derivative: parameter - learning_rate * derivative, parameters, derivatives)

    return parameters

def train():

    key, weight, bias, inputs, genuines = setup()
    parameters = init(key)

    begin = time.time()

    for i in range(5000):

        parameters = optimizer(parameters, inputs, genuines)

        if (i + 1) % 500 == 0:

            loss = loss_function(parameters, inputs, genuines)
            end = time.time()

            print(f"%.12fs is consumed while iterating" % (end - begin), f"%i times," % (i + 1), f"the loss now is %.12f" % loss)

            begin = time.time()

    return weight, bias, parameters

def start():

    weight, bias, parameters = train()

    input_test = jnp.array([0.17])
    genuine = weight * input_test + bias

    prediction = model(parameters, input_test)

    print("Final paramters: {}".format(parameters))
    print("input_test: {}, genuine: {}, predicted: {}".format(input_test, genuine, prediction))

def main():

    start()

if __name__ == "__main__":

    main()
