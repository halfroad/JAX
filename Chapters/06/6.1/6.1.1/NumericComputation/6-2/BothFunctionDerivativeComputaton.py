import jax


def body_function(x):

    return x ** 2

def start():

    grad_body_function = jax.value_and_grad(body_function)

    result = grad_body_function(1.0)

    print(result)

if __name__ == '__main__':

    start()
