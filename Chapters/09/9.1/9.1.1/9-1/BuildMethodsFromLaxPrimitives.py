import jax
from jax import lax
from jax._src import api


def multiply_add_lax(x, y, z):

    # Utilize the primitive method by Lax
    return lax.add(lax.mul(x, y), z)

def square_add_lax(a, b):

    # Custom method
    return multiply_add_lax(a, a, b)

def start():

    # Calculus by grad
    print("square_add_lax = ", square_add_lax(2., 10.))

    api_grad_square_add_lax = api.grad(square_add_lax, argnums = [0])
    print("lax.grad(square_add_lax) = ", api_grad_square_add_lax(2., 10.))

    grad_square_add_lax = jax.grad(square_add_lax, argnums = [0, 1])
    print("jax.grad(square_add_lax) = ", grad_square_add_lax(2., 10.))

def main():

    start()

if __name__ == "__main__":

    main()
