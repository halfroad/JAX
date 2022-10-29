import jax.random

"""

Paragraph 2.1.3, Fully Connected Layer with More functions
Program 2-2
Page 19
File Name: ExternelParametersLoadableFullyConnectedLayer.py

"""

def Dense(shape = [2, 1]):

    key = jax.random.PRNGKey(0)

    weights = jax.random.normal(key, shape = shape)
    biases = jax.random.normal(key, shape = (shape[-1],))

    _parameters = [weights, biases]

    def init_parameters():

        return _parameters

    def apply_func(inputs, parameters = _parameters):

        weight, bias = parameters

        return jax.numpy.dot(inputs, weight) + bias

    return init_parameters, apply_func

def start():

    key = jax.random.PRNGKey(17)
    shape = [2, 1]

    weights = jax.random.normal(key, shape = shape)
    biases = jax.random.normal(key, shape = (shape[-1],))

    parameters = [weights, biases]

    matrix = jax.numpy.array([

        [1.7, 1.7],
        [2.14, 2.14]
    ])

    init_parameters, apply_func = Dense()
    result = apply_func(matrix, parameters)

    print(f"result = {result}")

if __name__ == "__main__":

    start()
