import jax


def function(x):

    if x < 3:
        return 3. * x ** 2
    else:
        return -4 * x


def start():

    grad_function = jax.grad(function)

    # grad(3x^2) = 6x
    print(grad_function(2.))

    # grad(-4x) = -4
    print(grad_function(3.))

if __name__ == '__main__':

    start()
