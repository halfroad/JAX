import jax
import time


def setup():

    key = jax.random.PRNGKey(15)
    layers_shape = [1, 64, 128, 1]

    weight = 1.86
    bias = 1.68

    inputs = jax.random.normal(key = key, shape = layers_shape)
    genuines = inputs * weight + bias

    params = init_mlp_params(key, layers_shape)

    return (key, layers_shape), (weight, bias, params), (inputs, genuines)

def init_mlp_params(key, layers_shape):

    """

    this method is used to initialize the parameters

    """

    params = []

    # zip((1, 64, 128), (64, 128, 1)) -> (1, 64), (64, 128), (128, 1)
    for _in, _out in zip(layers_shape[: -1], layers_shape[1:]):
        #
        weight = jax.random.normal(key = key, shape = (_in, _out)) / 128.
        bias = jax.random.normal(key = key, shape = (_out, )) / 128.

        _dict = dict(weight = weight, bias = bias)

        params.append(_dict)

    return params

@jax.jit
def forward(params, inputs):

    # for param in params:
    #
    #     inputs = jax.numpy.matmul(inputs, param["weight"]) + param["bias"]
    #     inputs = jax.nn.selu(inputs)
    #
    # return inputs

    length = len(params)

    for i in range(length - 1):

        param = params[i]

        inputs = jax.numpy.matmul(inputs, param["weight"]) + param["bias"]
        inputs = jax.nn.selu(inputs)

    inputs = jax.numpy.matmul(inputs, params[-1]["weight"]) + params[-1]["bias"]

    return inputs

@jax.jit
def loss_function(params, inputs, genuines):

    predictions = forward(params, inputs)
    mse = jax.numpy.mean((predictions - genuines) ** 2)

    return mse

@jax.jit
def optimizer(params, inputs, genuines, learn_rate = 1e-1):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)

    params = jax.tree_util.tree_map(lambda param, gradient: param - gradient * learn_rate, params, gradients)

    return params

@jax.jit
def optimizer_v1(params, inputs, genuines, learn_rate = 1e-1):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)

    new_params = []

    for param, gradient in zip(params, gradients):

        new_weight = param["weight"] - gradient["weight"] * learn_rate
        new_bias = param["bias"] - gradient["bias"] * learn_rate

        _dict = dict(weight = new_weight, bias = new_bias)

        new_params.append(_dict)

    return new_params

def train():

    (key, layers_shape), (weight, bias, params), (inputs, genuines) = setup()

    start = time.time()

    for i in range(4000):

        params = optimizer_v1(params, inputs, genuines)

        if (i + 1) % 100 == 0:

            loss = loss_function(params, inputs, genuines)

            end = time.time()

            print("Time consumed while iterating: %.12fs," % (end - start), f"{i + 1} epoches completed, now the loss is {loss}")

            start = time.time()

    tests = jax.numpy.array([0.53])

    print(f"Genuine computed: {weight * tests + bias}")
    print(f"Computed after the model fitted: {forward(params, tests)}")

def main():

    train()

if __name__ == '__main__':

    main()
