import jax.lax


def add_func(i, x):

    return i + 1., x + 1.

def start():

    result = jax.lax.scan(add_func, 0, jax.numpy.array([1, 2, 3, 4]))

    print("result = ", result)

if __name__ == '__main__':

    start()
