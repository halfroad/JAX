import jax
import jax.numpy as jnp

def body_function(x):

    return x ** 2

def start():

    value_and_grad_function = jax.value_and_grad(body_function)

    result = value_and_grad_function(10.)

    print(result)

def main():

    start()

if __name__ == "__main__":

    main()
