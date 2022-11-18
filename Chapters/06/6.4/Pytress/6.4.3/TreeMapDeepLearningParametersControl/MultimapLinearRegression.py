import jax


def setup():

    """

    Dataset Preparation

    """

    prng = jax.random.PRNGKey(15)

    # Be noted the dimension of generated dataset is different from the dataset in paragraph 6.1.3
    shape = (1000, 1)

    inputs = jax.random.normal(key = prng, shape = shape)

    weight = 0.929
    bias = 0.214

    genuines = inputs * weight + bias

    layers_shape = [1, 64, 128, 1]

    return (prng, layers_shape, shape), (weight, bias), (inputs, genuines)

def init_mlp_params(layers_shape, key):

    params = []

    for _in, _out in zip (layers_shape[: -1], layers_shape[1:]):

        weight = jax.random.normal(key = key, shape = (_in, _out))
        bias = jax.random.normal(key = key, shape = (_out,))

        _dict = dict(weight = weight, bias = bias)

        params.append(_dict)

    return params

def forward(params, inputs):

    """

    Function of Multiple Layers Computation

    """

    for param in params:

        inputs = jax.numpy.matmul(inputs, param["weight"] + param["bias"])

    return inputs

def loss_function(params, inputs, genuines):

    """

    Loss Function

    """

    prediction = forward(params, inputs)
    loss = jax.numpy.square(jax.numpy.multiply(genuines, prediction))

    mean = jax.numpy.mean(loss)

    return mean

def optimizer(params, inputs, genuines, learn_rate = 1e-3):

    """

    Optimizer Function

    """

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)

    params = params - learn_rate * gradients

    return params

def optimizer_v1(params, inputs, genuines, learn_rate = 1e-3):

    """

    Optimizer Function

    """

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)

    # params = params - learn_rate * gradients

    return gradient

def optimizer_v2(params, inputs, genuines, learn_rate = 1e-3):

    """

    Optimizer Function

    """

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)

    new_params = []

    for param, gradient in zip(params, gradients):

        new_weight = param["weight"] - learn_rate * gradient["weight"]
        new_bias = param["weight"] - learn_rate * gradient["bias"]

        _dict = dict(weight = new_weight, bias = new_bias)

        new_params.append(_dict)

    return new_params

def optimizer_v3(params, inputs, genuines, learn_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)

    params = jax.tree_util.tree_map(lambda param, gradient: param - learn_rate * gradient, params, gradients)

    return params

def train():

    (prng, layers_shape, shape), (weight, bias), (inputs, genuines) = setup()
    params = init_mlp_params(layers_shape = layers_shape, key = prng)

    keys = jax.tree_util.tree_map(lambda x: x.shape, params)

    print("Keys = ", keys)

    gradients = optimizer_v3(params, inputs, genuines)

    for gradient in gradients:

        print("------------------------")

        weight = jax.numpy.array(gradient["weight"])
        bias = jax.numpy.array(gradient["bias"])

        print(weight.shape)
        print(bias.shape)

if __name__ == '__main__':

    train()
