import jax
import jax.numpy as jnp

from jax import random

def function(x):

    if x < 3:
        return 3. * x ** 2
    else:
        return -4 * x

def create_array(length, value):

    return jnp.ones((length, )) * value

def start():

    gradient_fucntion = jax.grad(function)

    """
    
    f(x) = 3 * x^2, grad(f(x)) = 3 * 2 * x
    g(x) = -4x, grad(g(x)) = -4 * 1 = -4
    
    """

    print(gradient_fucntion(2.))
    print(gradient_fucntion(3.))

    jit_function = jax.jit(function)

    # Error, x is not concrete
    # print(jit_function(2.))

    # Solution: Explicit to let jit know the type of parameters
    # static_argnums = (0,), 0 means 0 dimension - single parameter
    jit_function = jax.jit(function, static_argnums = (0,))

    print(jit_function(2.))

    array = create_array(5, 4)

    print(array)

    jit_create_array = jax.jit(create_array)

    # Error. Input (5, 4) is integer, but concrete abstract tracer is expected
    # array = jit_create_array(5, 4)

    # print(array)

    jit_create_array = jax.jit(create_array, static_argnums = (0, ))

    array = jit_create_array(5, 4)

    print(array)

def main():

    start()


if __name__ == "__main__":

    main()
