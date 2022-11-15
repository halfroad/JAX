import jax


def body_function(x, y):

    return x * y

def start():

    grad_body_function = jax.grad(body_function)

    x = 2.
    y = 3.

    result = grad_body_function(x, y)

    print(result)

if __name__ == '__main__':

    start()

