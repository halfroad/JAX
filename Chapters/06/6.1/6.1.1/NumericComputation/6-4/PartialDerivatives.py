import jax


def body_function(x, y):

    return x * y

def start():

    grad_body_function = jax.grad(body_function)

    x = 2.
    y = 3.

    grad_body_function = jax.grad(body_function, argnums = (0, 1))

    dx, dy = grad_body_function(x, y)

    print("dx = {}, dy = {}".format(dx, dy))

if __name__ == '__main__':

    start()
