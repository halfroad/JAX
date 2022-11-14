import jax


def function(x):

    if x < 3:
        return 3. * x ** 2
    else:
        return -4 * x

def start():

    # jit_function = jax.jit(function)

    # grad(3x^2) = 6x
    # print(jit_function(2.))

    jit_function = jax.jit(function, static_argnums = (0,))

    print(jit_function(2.))



if __name__ == '__main__':

    start()
