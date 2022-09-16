import jax
import jax.numpy as jnp

def aggregate(x):

    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

def derivative(x):

    #return jnp.exp(-x) / ((1 + jnp.exp(-x)) ** 2)
    return jnp.exp(-x) / jnp.power(1 + jnp.exp(-x), 2)

def start():

    x = jnp.arange(3.)

    print(x)

    derivative_function = jax.grad(aggregate)

    x_derivative = derivative_function(x)

    print(x_derivative)

    x_derivative = derivative(x)

    print(x_derivative)

def main():

    start()

if __name__ == "__main__":

    main()
