import jax


def immutable_array():

    array = jax.numpy.linspace(0, 9, 10)

    print(type(array))

    array[0] = 17
    print(array)

def start():

    immutable_array()

if __name__ == "__main__":

    start()
