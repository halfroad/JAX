import jax
import jax.numpy as jnp

def body_function(x, y, z):

    return x * y * z

def start():

    grad_body_function = jax.grad(body_function)

    x = 2.
    y = 3.
    z = 4.

    # argnums mean the parameter positions to grad
    grad_body_function = jax.grad(body_function, argnums = (0, 1, 2))

    dx, dy, dz = grad_body_function(x, y, z)

    print(f"dx = {dx}, dy = {dy}, dz = {dz}")

    # Error
    # grad_body_function = jax.grad(body_function, argnums = (0, 1, 2, 3))

    # dx, dy, dz = grad_body_function(x, y, z)

def main():

    start()

if __name__ == "__main__":

    main()
