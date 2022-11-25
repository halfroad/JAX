import jax


@jax.custom_jvp
def function(x):

    return 2 * x

@function.defjvp
def function_jvp(primals, tangents):

    x, = primals
    x_dot, = tangents

    if x >= 0:

        return function(x), x_dot

    else:

        return function(x), 2 * x_dot

def run():

    grad_function = jax.grad(function)

    # Derivate the x_dot
    print("grad_function(1.) = ", grad_function(1.))

    # Derivate the 2 * x_dot
    print("grad_function(-1.) = ", grad_function(-1.))

if __name__ == '__main__':

    run()
