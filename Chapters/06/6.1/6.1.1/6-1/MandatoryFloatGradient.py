import jax
import jax.numpy as jnp

def body_function(x):

    return x ** 2

def main():

    grad_body_function = jax.grad(body_function)

    # Error
    result = grad_body_function(1.)

    print(result)

if __name__ == "__main__":

    main()
