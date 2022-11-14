import jax.numpy


def update_array():

    array = jax.numpy.zeros((3, 3), dtype = jax.numpy.float32)

    print(array)

    array[1, :] = 1.0

if __name__ == '__main__':

    update_array()
