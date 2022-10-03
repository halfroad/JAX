import jax
from jax import custom_vjp


@custom_vjp
def function(x, y):

    return x * y

def functio_forwards(x, y):

    # Define the forwards computing function and the derivative function
    return function(x, y), (y, x)

def function_backwards(res, g):

    # Define the derivative result
    y, x = res

    return y, x

def start():

    # Register the forwards and backwards derivative functions in the custom function
    function.defvjp(functio_forwards, function_backwards)

    grad_function = jax.grad(function)
    print("jax.grad(function) =", grad_function(2., 3.))

    grad_function = jax.grad(function, [0, 1])
    print("jax.grad(function) =", grad_function(2., 3.))

def main():

    start()

if __name__ == "__main__":

    main()
