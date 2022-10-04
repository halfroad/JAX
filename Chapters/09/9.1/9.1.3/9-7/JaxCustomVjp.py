import jax
from jax import custom_vjp


@custom_vjp
def function(x):

    return x ** 2

def function_forwards(x):

    # original function and derivative function
    return function(x), (2 * x,)

def funciton_backwards(res, g):

    # Reverse derivative function
    dot_x = res

    return (g,)

def start():

    function.defvjp(function_forwards, funciton_backwards)

    grad_function = jax.grad(function)

    print(grad_function(3.))


def main():

    start()

if __name__ == "__main__":

    main()
