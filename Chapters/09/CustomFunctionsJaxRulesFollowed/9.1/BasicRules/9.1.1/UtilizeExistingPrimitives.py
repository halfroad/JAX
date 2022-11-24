import jax
from jax._src import api


def multiply_add_lax(x, y, z):

    # Utilize the exsiting function in jax.lax
    multiplication = jax.lax.mul(x, y)
    addition = jax.lax.add(multiplication, z)

    return addition

def square_add_lax(a, b):

    # Use custom function
    return multiply_add_lax(a, a, b)

def test():

    # Utilize the grad to compute the derivative of function
    squared_add_lax = square_add_lax(2., 10.)
    print("squared_add_lax = ", squared_add_lax)

    print("---------------------")

    grad_square_add_lax = api.grad(square_add_lax, argnums = [0])
    print("grad_square_add_lax = ", grad_square_add_lax(2.0, 10.))

    print("---------------------")

    grad_square_add_lax = api.grad(square_add_lax, argnums = [0, 1])
    print("grad_square_add_lax = ", grad_square_add_lax(2.0, 10.))

if __name__ == '__main__':

    test()
