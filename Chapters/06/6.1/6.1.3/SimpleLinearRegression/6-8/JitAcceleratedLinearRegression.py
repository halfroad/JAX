import time

import jax


def setup():

    prng = jax.random.PRNGKey(15)

    shape = (1000,)
    inputs = jax.random.normal(prng, shape = shape)

    weight = 0.929
    bias = 0.214

    genuines = inputs * weight + bias

    params = jax.random.normal(key = prng, shape = (2,))

    print(f"The initial params = {params}")

    return (prng, shape), (weight, bias, params), (inputs, genuines)

# Attributed by jax.jit
@jax.jit
def model(params, inputs):

    weight = params[0]
    bias = params[1]

    prediction = inputs * weight + bias

    return prediction

@jax.jit
def loss_function(params, inputs, genuines):

    prediction = model(params, inputs)
    mse = jax.numpy.mean((prediction - genuines) ** 2)

    return mse

@jax.jit
def update(params, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradient = grad_loss_function(params, inputs, genuines)

    # Update the params
    params = params - gradient * learning_rate

    return params

def train():

    (prng, shape), (weight, bias, params), (inputs, genuines) = setup()

    start = time.time()
    start_ = start

    for i in range(4000):

        params = update(params, inputs, genuines)

        if (i + 1) % 500 == 0:

            loss = loss_function(params, inputs, genuines)

            end = time.time()

            print(f"Time {end - start}s is consumed while iterating epoch {i + 1}, the loss now is {loss}")

            start = time.time()

    print(f"The final params = {params}, total time consumed {end - start_}")


if __name__ == '__main__':

    train()
