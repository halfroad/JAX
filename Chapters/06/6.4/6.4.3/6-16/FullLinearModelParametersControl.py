import jax
import jax.numpy as jnp
import time

def setup():

    key = jax.random.PRNGKey(17)
    inputs = jax.random.normal(key, shape = (1000, 1))

    weight = .929
    bias = .214

    genuines = weight * inputs + bias

    layers_shape = [1, 64, 128, 1]

    return key, weight, bias, layers_shape, inputs, genuines

def init_multilayer_perceptron_parameters(key, layers_shape):

    initial_paramters = []

    # layers_shape = [1, 64, 128, 1],
    # layers_shape[: -1] = [1, 64, 128],
    # layers_shape[1:] = [64, 128, 1]
    for _in, _out in zip(layers_shape[: -1], layers_shape[1:]):

        weight = jax.random.normal(key, shape = (_in, _out)) / 128.
        bias = jax.random.normal(key, shape = (_out,)) / 128.

        _dict = dict(weight = weight, bias = bias)

        initial_paramters.append(_dict)

    return initial_paramters

@jax.jit
def forward(paramters, inputs):

    for paramter in paramters:

        inputs = jnp.matmul(inputs, paramter["weight"]) + paramter["bias"]

    return inputs

@jax.jit
def loss_function(paramters, inputs, genuines):

    predictions = forward(paramters, inputs)
    differences = (predictions - genuines) ** 2

    return jnp.mean(differences)

@jax.jit
def optimizer_function(paramters, inputs, genuines, learning_rate = 1e-1):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(paramters, inputs, genuines)

    mapped_parameters = jax.tree_util.tree_map(lambda parameter, gradient: parameter - learning_rate * gradient, paramters, gradients)

    return mapped_parameters

# Recommanded optimizer
@jax.jit
def optimzier_function_v2(paramters, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(paramters, inputs, genuines)

    new_parameters = []

    for paramter, gradient in zip(paramters, gradients):

        new_weight = paramter["weight"] - learning_rate * gradient["weight"]
        new_bias = paramter["bias"] - learning_rate * gradient["bias"]

        _dict = dict(weight = new_weight, bias = new_bias)

        new_parameters.append(_dict)

    return new_parameters

def start():

    key, weight, bias, layers_shape, inputs, genuines = setup()
    parameters = init_multilayer_perceptron_parameters(key, layers_shape)
    mapped_initial_parameters = jax.tree_util.tree_map(lambda x: x.shape, parameters)

    print("initial_parameters = {}, mapped_initial_parameters = {}".format(parameters, mapped_initial_parameters))

    begin = time.time()

    for i in range(40000):

        parameters = optimzier_function_v2(parameters, inputs, genuines)

        if (i + 1) % 500 == 0:

            loss = loss_function(parameters, inputs, genuines)
            end = time.time()

            print(f"%.12fs" % (end - begin), f"is consumed while iterating %i," % (i + 1), f"now the loss is %.12f" % loss)

            begin = time.time()

    inputs_test = jnp.array([.17])
    genuine = weight * inputs_test + bias
    prediction = forward(parameters, inputs_test)

    print("the genuine computed: {}, predicted: {}".format(genuine, prediction))

def main():

    start()

if __name__ == "__main__":

    main()
