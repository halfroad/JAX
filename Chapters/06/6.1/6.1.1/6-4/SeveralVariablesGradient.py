import jax
import jax.numpy as jnp

def body_function(x, y):

    return (x ** 2) * (y ** 4)

def start():

    grad_function = jax.grad(body_function)

    x = 2.
    y = 3.

    result = grad_function(x, y)

    grad_function = jax.grad(body_function, argnums = (0, 1))

    dx, dy = grad_function(x, y)

    print(f"dx: {dx}")
    print(f"dy: {dy}")

def main():

    start()

if __name__ == "__main__":

    main()
