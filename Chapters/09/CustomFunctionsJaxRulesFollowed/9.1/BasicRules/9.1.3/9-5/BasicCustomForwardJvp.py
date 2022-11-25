import jax


@jax.custom_jvp
def function(x):

    return x

def function_jvp(primals, tangents):

    x, = primals
    t, = tangents

    return function(x), t * x

@jax.custom_jvp
def function(x):

    return x

def function_jvp(primals, tangents):

    x, = primals
    t, = tangents

    return x * t

def run():

    function.defjvp(function_jvp)

    print("function(3.) = ", function(3.))

    grad_function = jax.grad(function)
    print("grad_function(2.) = ", grad_function(2.))

    y, y_dot = jax.jvp(function, (3.,), (2.,))

    # Print the computing result of function and custom derivative (t * x) on function_jvp
    print("y = ", y, "y_dot = ", y_dot)

if __name__ == '__main__':

    run()

