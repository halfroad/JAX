import time

import jax


def setup():

    key = jax.random.PRNGKey(15)
    input_shape = (1000,)

    inputs = jax.random.normal(key, shape = input_shape)

    weight = 0.929
    bias = 0.214

    # Initial params
    params = jax.random.normal(key, (2, ))

    genuines = inputs * weight + bias

    return (key, input_shape), (weight, bias), params, (inputs, genuines)

def model(params, inputs):

    """

    Create the model

    """

    weight = params[0]
    bias = params[1]

    predictions = inputs * weight + bias

    return predictions

def loss_fucntion(params, inputs, genuines):

    prediction = model(params, inputs)
    loss = jax.numpy.mean((prediction - genuines) ** 2)

    return loss

def update(params, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_fucntion)
    gradient = grad_loss_function(params, inputs, genuines)

    params = params - learning_rate * gradient

    return params

def train():

    (key, input_shape), (weight, bias), params, (inputs, genuines) = setup()

    print(f"The initial params is {params}")

    begin = time.time()

    for i in range(4000):

        params = update(params, inputs, genuines)

        if (i + 1) % 100 == 0:

            loss = loss_fucntion(params, inputs, genuines)

            end = time.time()

            print(f"%.2fs is consumed" % (end - begin), f"while iterating epoch {i + 1},", f"the loss now is {loss}")

            begin = time.time()

    print(f"The final params is {params}")

if __name__ == '__main__':

    train()
