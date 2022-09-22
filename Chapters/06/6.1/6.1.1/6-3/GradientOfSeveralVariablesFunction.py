import jax
import jax.numpy as jnp

def body_function(x, y):

    return x * y

def start():

    grad_function = jax.grad(body_function)

    x = 2.
    y = 3.

    result = grad_function(x, y)

    print(result)

def main():

    start()

if __name__ == "__main__":

    main()
