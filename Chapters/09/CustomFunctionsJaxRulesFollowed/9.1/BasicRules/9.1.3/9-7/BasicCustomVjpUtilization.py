import jax


@jax.custom_vjp
def function(x):

    # Custom function
    return x ** 2

def function_fwd(x):

    # Here the original function and manual computation of derivative returned
    return function(x), 2 * x

def function_bwd(dot_x, y_bar):

    return (dot_x, )

def run():

    function.defvjp(function_fwd, function_bwd)

    grad_function = jax.grad(function)

    print("grad_function(3.) = ", grad_function(3.))

if __name__ == '__main__':

    run()


