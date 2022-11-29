import jax


def no_bounary_check():

    array = jax.numpy.arange(10)

    print(array)
    print(array[17])

if __name__ == '__main__':

    no_bounary_check()
