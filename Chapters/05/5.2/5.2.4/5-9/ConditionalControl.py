from jax import lax

import jax.numpy as jnp

def add(x):

    return x + 1.

def subtract(x):

    return x - 1.

def cond_function(x):

    return x < 17

def body_function(x):

    return x + 1

def for_loop_body_function(i, x):

    return x + 1

def scan_function(i, x):

    return i + 1., x + 1.

def start():

    operand = jnp.array([10.])

    # lambda is anonymous method
    result = lax.cond(True, lambda x: x + 1, lambda x: x - 1, operand)

    print("Result = ", result)
    print("------------------------------------")

    result = lax.cond(True, add, subtract, operand)

    print("Result = ", result)
    print("------------------------------------")

    operand = jnp.array([-10.])

    result = lax.cond(False, lambda x: x + 1, lambda x: x - 1, operand)

    print("Result = ", result)
    print("------------------------------------")

    result = lax.cond(False, add, subtract, operand)

    print("Result = ", result)
    print("------------------------------------")

    operand = 100.

    # lambda is anonymous method
    result = lax.cond(operand > 0, lambda x: x + 1, lambda x: x - 1, operand)

    print("Result = ", result)
    print("------------------------------------")

    result = lax.cond(operand > 0, add, subtract, operand)

    print("Result = ", result)
    print("------------------------------------")

    operand = -100.

    result = lax.cond(operand <= 0, lambda x: x + 1, lambda x: x - 1, operand)

    print("Result = ", result)
    print("------------------------------------")

    result = lax.cond(operand <= 0, add, subtract, operand)

    print("Result = ", result)
    print("------------------------------------")

    initial = 1

    result = lax.while_loop(cond_function, body_function, init_val = 1)

    print("Result = ", result)
    print("------------------------------------")

    initial = 1
    begin = 0
    stop = 100

    for_loop_body_handler = lambda i, x: x + i + 1
    result = lax.fori_loop(begin, stop, for_loop_body_handler, initial)

    print("Result = ", result)
    print("------------------------------------")

    result = lax.fori_loop(begin, stop, for_loop_body_function, initial)

    print("Result = ", result)
    print("------------------------------------")

    result = lax.scan(scan_function, 0, jnp.array([1, 2, 3, 4]))

    print("Result = ", result)
    print("------------------------------------")


def main():

    start()

if __name__ == "__main__":

    main()
