import jax
import jax.numpy as jnp

def simple_linear_equation():

    key = jax.random.PRNGKey(17)

    x = jax.random.normal(key, shape = (1000,))

    a = 0.929
    b = 0.214

    y = a * x + b

    return y

def setup():

    key = jax.random.PRNGKey(17)

    parameters = jax.random.normal(key, shape = (2,))

    return parameters

# Build the Linear Regression Model
def model(parameters, input_):

    a = parameters[0]    # Obtain parameters
    b = parameters[1]

    y = a * input_ + b

    return y

# Create the function to compute the loss
def loss_function(parameters, input_, genuine):

    prediction = model(parameters, input_)

    mean_squared_error = jnp.mean((prediction - genuine) ** 2)

    return mean_squared_error

def update(parameters, input_, genuine, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    loss = grad_loss_function(parameters, input_, genuine)

    return parameters - learning_rate * loss

def start():
    setup()

def main():

    start()

if __name__ == "__main__":

    main()
