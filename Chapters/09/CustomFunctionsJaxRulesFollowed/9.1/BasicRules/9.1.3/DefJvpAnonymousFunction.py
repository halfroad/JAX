import jax


@jax.custom_jvp
def function(x):

    return 2 * x

def run():

    function.defjvps(lambda primals, tangents, t: primals)

    grad_function = jax.grad(function)

    print("grad_function(3.) = ", grad_function(3.))

if __name__ == '__main__':

    run()
