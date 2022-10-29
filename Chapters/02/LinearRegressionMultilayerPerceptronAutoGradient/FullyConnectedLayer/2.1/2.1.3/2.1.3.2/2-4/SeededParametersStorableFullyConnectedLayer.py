"""

Paragraph 2.1.3, Parameters Storable Fully Connected Layer
Program 2-4
Page 21

"""
import jax.random


def Dense(seed = 15, shape = [2, 1]):

    def init_function(input_shape = shape):

        key = jax.random.PRNGKey(seed)

        weights, biases = jax.random.normal(key, shape = input_shape), jax.random.normal(key, shape = (input_shape[-1],))

        return weights, biases

    def apply_function(inputs, parameters):

        weight, bias = parameters

        return jax.numpy.dot(inputs, weight) + bias

    return init_function, apply_function

def start():

    init_function, apply_function = Dense(seed = 17)
    parameters = init_function()

    matrix = jax.numpy.array([
        [1.7, 1.7],
        [2.14, 2.14]
    ])

    result = apply_function(inputs = matrix, parameters = parameters)

    print(f"result = {result}")

    init_function, apply_function = Dense(seed = 18)

    # Be noted, here the parameters created for seed 17 is reused
    result = apply_function(inputs = matrix, parameters = parameters)

    print(f"result = {result}")

if __name__ == "__main__":

    start()
