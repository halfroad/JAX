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

    layers_shape = [1,64, 128, 1]

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
    gradient = grad_loss_function(params, inputs, genuines)

    params = params - learn_rate * gradient

    return params


def train():

    (prng, layers_shape, shape), (weight, bias), (inputs, genuines) = setup()
    params = init_mlp_params(layers_shape = layers_shape, key = prng)

    keys = jax.tree_util.tree_map(lambda x: x.shape, params)

    print("Keys = ", keys)

if __name__ == '__main__':

    train()
