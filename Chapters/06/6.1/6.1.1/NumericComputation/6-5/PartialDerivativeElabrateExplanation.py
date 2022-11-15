import jax


def body_function(x, y, z):

    return x * y * z

def start():

    grad_body_function = jax.grad(body_function)

    x = 2.
    y = 3.
    z = 4.

    # grad_body_function = jax.grad(body_function, argnums = (0, 1, 2))

    # Error
    grad_body_function = jax.grad(body_function, argnums = (0, 1, 2, 3))


    result = grad_body_function(x, y, z)

    print(result)

if __name__ == '__main__':

    start()
