import jax


def body_function(x):

    return x ** 2

def start():

    grad_body_function = jax.grad(body_function)

    # This is not correct
    # print(grad_body_function(1))
    print(grad_body_function(1.))

if __name__ == '__main__':

    start()
