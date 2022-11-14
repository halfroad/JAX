import jax
from jax import lax


def add_function(x):

    return x + 1

def substract_function(x):

    return x - 1

def start():

    operand = 0
    
    result = lax.cond(operand > 0, add_function, substract_function, operand)
    print("result = ", result)

    result = lax.cond(operand <= 0, add_function, substract_function, operand)
    print("result = ", result)


if __name__ == '__main__':

    start()
