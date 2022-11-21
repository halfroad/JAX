import jax


def start():

    matrix1 = jax.numpy.arange(100)
    print(matrix1.shape)

    kernel = jax.numpy.arange(9)
    print(kernel.shape)

    result = jax.numpy.convolve(matrix1, kernel)

    print(result.shape)

if __name__ == '__main__':

    start()
