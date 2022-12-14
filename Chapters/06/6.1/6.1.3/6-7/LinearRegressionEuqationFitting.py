import time
import jax
import jax.numpy as jnp

import inspect

def setup():

    print(inspect.currentframe().f_code.co_name)

    key = jax.random.PRNGKey(17)

    inputs = jax.random.normal(key, shape = (1000,))

    weight = .929
    bias = .214

    genuines = weight * inputs + bias

    # Randomly create the parameters, find weight and bias by Linear Regression and Gradient
    parameters = jax.random.normal(key, shape = (2,))

    print("Initial Parameter: ", parameters)

    return inputs, genuines, parameters

def model(parameters, inputs):

    weight = parameters[0]
    bias = parameters[1]

    predictions = weight * inputs + bias

    return predictions

def loss_function(parameters, inputs, genuines):

    predictions = model(parameters, inputs)

    return jnp.mean((predictions - genuines) ** 2)

def update(paramters, inputs, genuines, learning_rate = 1e-3):

    # Be noted that  the inputs and genuines are always the same datasets, but try with gradient Weight and Bias
    grad_loss_function = jax.grad(loss_function)
    gradient_losses = grad_loss_function(paramters, inputs, genuines)

    new_paramters = paramters - learning_rate * gradient_losses

    return new_paramters

def start():

    print(inspect.currentframe().f_code.co_name)

    begin = time.time()

    inputs, genuines, parameters = setup()

    for i in range(4000):

        parameters = update(parameters, inputs, genuines)

        if (i + 1) % 500 == 0:

            loss = loss_function(parameters, inputs, genuines)
            end = time.time()

            weight = parameters[0]
            bias = parameters[1]

            print("%.12fs is consumed" % (end - begin), f"when iterarting number %i," % (i + 1), f"the loss is %.12f" % loss, f"the weight is %.12f," % weight, f"the bias is %.12f" % bias)

            begin = time.time()

    print("The final weight and bias: ", parameters)

def main():

    start()

if __name__ == "__main__":

    main()
