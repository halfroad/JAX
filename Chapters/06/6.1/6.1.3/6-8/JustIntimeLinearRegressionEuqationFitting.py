import time
import jax
import jax.numpy as jnp

@jax.jit
def setup():

    key = jax.random.PRNGKey(17)

    inputs = jax.random.normal(key, shape = (1000,))

    weight = .929
    bias = .214

    genuines = inputs * weight + bias

    # Randomly create the parameters, find weight and bias by Linear Regression and Gradient
    parameters = jax.random.normal(key, shape = (2,))

    print("Initial Parameter: ", parameters)

    return parameters, inputs, genuines

@jax.jit
def model(parameters, inputs):

    weight = parameters[0]
    bias = parameters[1]

    predictions = weight * inputs + bias

    return predictions

@jax.jit
def loss_function(parameters, inputs, genuines):

    predictions = model(parameters, inputs)

    # Only 1 value for loss, not dataset [losses]
    return jnp.mean((predictions - genuines) ** 2)

@jax.jit
def update(parameters, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    loss = grad_loss_function(parameters, inputs, genuines)

    new_parameters = parameters - learning_rate * loss

    return new_parameters

def start():

    parameters, inputs, genuines = setup()

    begin = time.time()

    for i in range(4000):

        parameters = update(parameters, inputs, genuines)

        if (i + 1) % 500 == 0:

            loss = loss_function(parameters, inputs, genuines)
            end = time.time()

            weight = parameters[0]
            bias = parameters[1]

            print("%.12fs is consumed" % (end - begin), f"when iterarting number %i," % (i + 1), f"the loss is %.12f" % loss, f"the weight is %.12f," % weight, f"the bias is %.12f" % bias)

    print("The final weight and bias: ", parameters)

def main():

    start()

if __name__ == "__main__":

    main()
