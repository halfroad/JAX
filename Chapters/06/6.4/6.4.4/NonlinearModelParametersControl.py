import jax.nn
import jax.numpy as jnp
import time

def setup():

    key = jax.random.PRNGKey(17)
    inputs = jax.random.normal(key, shape = (1000, 1))

    weight = .929
    bias = .214

    genuines = weight * inputs + bias

    layers_shape = [1, 64, 128, 1]

    return key, weight, bias, inputs, genuines, layers_shape

def init_multilayers_perceptron_parameters(key, layers_shape):

    parameters = []

    for begin, end in zip(layers_shape[: -1], layers_shape[1:]):

        weight = jax.random.normal(key, shape = (begin, end)) / 128.
        bias = jax.random.normal(key, shape = (end,)) / 128.

        _dict = dict(weight = weight, bias = bias)

        parameters.append(_dict)

    return parameters

@jax.jit
def forward(parameters, inputs):

    length = len(parameters)

    for i in range(length - 1):

        parameter = parameters[i]

        inputs = jnp.matmul(inputs, parameter["weight"]) + parameter["bias"]
        inputs = jax.nn.selu(inputs)

    inputs = jnp.matmul(inputs, parameters[-1]["weight"]) + parameters[-1]["bias"]

    return inputs

@jax.jit
def loss_function(parameters, inputs, genuines):

    predictions = forward(parameters, inputs)
    differences = (genuines - predictions) ** 2

    return jnp.mean(differences)

# Workable, but not recommanded
@jax.jit
def optimizer_function(parameters, inputs, genuines, learning_rate = 1e-1):

    grad_loss_function = jax.grad(loss_function)
    derivatives = grad_loss_function(parameters, inputs, genuines)

    new_parameters = jax.tree_util.tree_map(lambda parameter, gradient: parameter - learning_rate * gradient, parameters, derivatives)

    return new_parameters

# Recommanded solution
@jax.jit
def optimizer_funciton_v2(parameters, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    derivatives = grad_loss_function(parameters, inputs, genuines)

    new_parameters = []

    for parameter, derivative in zip(parameters, derivatives):

        weight = parameter["weight"] - learning_rate * derivative["weight"]
        bias = parameter["bias"] - learning_rate * derivative["bias"]

        _dict = dict(weight = weight, bias = bias)

        new_parameters.append(_dict)

    return new_parameters

def train():

    key, weight, bias, inputs, genuines, layers_shape = setup()
    parameters = init_multilayers_perceptron_parameters(key, layers_shape)

    begin = time.time()

    for i in range(60000):

        parameters = optimizer_funciton_v2(parameters, inputs, genuines)

        if (i + 1) % 500 == 0:

            loss = loss_function(parameters, inputs, genuines)

            end = time.time()

            print(f"%.12fs" % (end - begin), f"is consumed while iterating %i times," % (i + 1), f"now loss is = %s" % loss)

            begin = time.time()

    return weight, bias, parameters

def start():

    weight, bias, parameters = train()

    inputs_test = jnp.array([.17])
    genuine = weight * inputs_test + bias

    prediction = forward(parameters, inputs_test)

    print("the genuine computed: {}, predicted: {}".format(genuine, prediction))

def main():

    start()

if __name__ == "__main__":

    main()
