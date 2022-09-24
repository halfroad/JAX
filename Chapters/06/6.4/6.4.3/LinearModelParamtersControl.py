import jax
import jax.numpy as jnp

def setup():

    key = jax.random.PRNGKey(17)

    # Here the dimensions is different from paragraph 6.1.3
    inputs = jax.random.normal(key, shape = (1000, 1))

    weight = .929
    bias = .214

    genuines = weight * inputs + bias

    layers_shape = [1, 64, 128, 1]

    return inputs, genuines, key, layers_shape

def init_multilayer_perceptron_parameters(key, layers_shape):

    """

    1. input layer (1 dimension) -> hidden layer (64 dimensions) -> hidden layer (128 dimensions) -> output layer (1 dimension)

    """
    parameters = []

    slices1 = layers_shape[: -1]
    slices2 = layers_shape[1:]
    combination = zip(slices1, slices2)

    for _in, _out in combination:

        weights = jax.random.normal(key, shape = (_in, _out))
        biases = jax.random.normal(key, shape = (_out,))

        _dict = dict(weight = weights, bias = biases)

        parameters.append(_dict)

    return parameters

# Multilayers computing function
def forward(parameters, inputs):

    for parameter in parameters:

        inputs = jnp.matmul(inputs, parameter["weight"]) + parameter["bias"]

    return inputs

# Loss function
def loss_function(parameters, inputs, genuines):

    predictions = forward(parameters, inputs)
    losses = jnp.square(jnp.multiply(genuines, predictions))

    return jnp.mean(losses)

# Optimizer function
def optimizer_function(parameters, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(parameters, inputs, genuines)

    parameters = parameters - learning_rate * gradients

    return grad_loss_function, parameters, gradients

# Can be utilized but not recommanded
def optimizer_function_v2(parameters, inputs, genuines, learning_rate = 1e-1):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(parameters, inputs, genuines)

    new_parameters = []

    for parameter, gradient in zip(parameters, gradients):

        new_weight = parameter["weight"] - learning_rate * gradient["weight"]
        new_bias = parameter["bias"] - learning_rate * gradient["bias"]

        _dict = dict(weight = new_weight, bias = new_bias)

        new_parameters.append(_dict)

    return grad_loss_function, new_parameters, gradients

# Recommanded operimzier function
def optimizer_function_v3(parameters, inputs, genuines, learning_rate = 1e-1):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(parameters, inputs, genuines)

    new_parameters = jax.tree_util.tree_map(lambda parameter, gradient: parameter - learning_rate * gradient, parameters, gradients)

    return grad_loss_function, new_parameters, gradients

def start():

    xs, ys, key, layers_shape = setup()

    parameters = init_multilayer_perceptron_parameters(key, layers_shape)
    mapped = jax.tree_util.tree_map(lambda x: x.shape, parameters)

    print(mapped)

    key = jax.random.PRNGKey(17)
    inputs = jax.random.normal(key, shape = (1000, 1))

    weight = .929
    bias = .214

    genuines = weight * inputs + bias

    grad_loss_function, parameters, gradients = optimizer_function_v3(parameters, inputs, genuines)

    print(gradients)

    for grd in gradients:

        print("---------------------------")

        weight = jnp.array(grd["weight"])
        bias = jnp.array(grd["bias"])

        print(weight.shape)
        print(bias.shape)

def main():

    start()

if __name__ == "__main__":

    main()
