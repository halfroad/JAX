import jax
import time


def setup():

    prng = jax.random.PRNGKey(15)
    layers_shape = [1, 64, 128, 1]

    weight = 0.716
    bias = 0.186

    inputs = jax.random.normal(prng, shape = layers_shape)
    genuines = inputs * weight + bias

    params = init_mlp_params(key = prng, layers_shape = layers_shape)

    return (layers_shape, prng), (weight, bias, params), (inputs, genuines)

def init_mlp_params(key, layers_shape):

    params = []

    for _in, _out in zip(layers_shape[: -1], layers_shape[1:]):

        weight = jax.random.normal(key, shape = (_in, _out)) / 128.
        bias = jax.random.normal(key, shape = (_out, )) / 128.0

        _dict = dict(weight = weight, bias = bias)

        params.append(_dict)

    return params

@jax.jit
def forward(params, inputs):

    for param in params:

        inputs = jax.numpy.matmul(inputs, param["weight"]) + param["bias"]

    return inputs

@jax.jit
def loss_function(params, inputs, genuines):

    prediction = forward(params, inputs)
    mse = jax.numpy.mean((prediction - genuines) ** 2)

    return mse

@jax.jit
def optimzier(params, inputs, genuines, learn_rate):

    """

    Optimizer of Stochastic Gradient Descent

    """

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)
    params = jax.tree_util.tree_map(lambda param, gradient: param - learn_rate * gradient, params, gradients)

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

    (layers_shape, prng), (weight, bias, params), (inputs, genuines) = setup()

    print(jax.tree_util.tree_map(lambda x: x.shape, params))

    start = time.time()

    for i in range(4000):

        params = optimizer_v1(params, inputs, genuines)

        if (i + 1) % 100 == 0:

            loss = loss_function(params, inputs, genuines)

            end = time.time()

            print(f"Time {end - start}s is consumed while iterating epoches {i + 1}, now the loss is {loss}")

            start = time.time()

    tests = jax.numpy.array([0.17])

    print(f"Genuine by computation: {weight * tests + bias}")
    print(f"Predicted after fitting the model: {forward(params, tests)}")

def main():

    train()

if __name__ == '__main__':

    main()
