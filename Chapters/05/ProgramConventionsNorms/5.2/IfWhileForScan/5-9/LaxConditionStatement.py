import jax.numpy
from jax import lax


def add_function(x):

    return x + 1.

def subtract_function(x):

    return x - 1.

def start():

    operand = jax.numpy.array([0.1])

    result = lax.cond(True, lambda x: x + 1, lambda x: x - 1, operand)
    print("result = ", result)

    result = lax.cond(False, lambda x: x + 1, lambda x: x - 1, operand)
    print("result = ", result)

    print("-------------------------")

    result = lax.cond(True, add_function, subtract_function, operand)
    print("result = ", result)

    result = lax.cond(False, add_function, subtract_function, operand)
    print("result = ", result)


if __name__ == '__main__':

    start()
