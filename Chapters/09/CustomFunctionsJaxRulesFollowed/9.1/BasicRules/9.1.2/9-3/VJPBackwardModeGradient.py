import jax


@jax.custom_vjp

def function(x, y):

    return x * y

def function_forward(x, y):

    # Define the forward computing function and the derivative funcitons
    return function(x, y), (y, x)

def function_backward(result, gradient):

    # Define the result of derivative
    y, x = result

    return y, x

def test():

    # Register the forward and backward derivative
    function.defvjp(function_forward, function_backward)

    grad_function = jax.grad(function)
    print("grad_function(2., 3.) = ", grad_function(2., 3.))

    grad_function = jax.grad(function, [0, 1])
    print("jax.grad(function, [0, 1])(2., 3.) = ", grad_function(2., 3.))

if __name__ == '__main__':

    test()




