import jax.numpy


def boundary_check():

    array = jax.numpy.arange(9)

    print(array)
    print(array[-1])
    print(array[11])

def array_add_types_check():

    """

    Both types should be created in JAX for add operator on 2 arrays

    """

    # Right without error
    array = jax.numpy.arange(9)
    accumlation = jax.numpy.sum(array)

    print(accumlation)

    # Error
    array = range(9)
    accumlation = jax.numpy.sum(array)

    print(accumlation)

def wrap_by_jax_array_to_solve_inefficiency():

    array = jax.numpy.arange(9)
    array = jax.numpy.array(array)
    accumlation = jax.numpy.sum(array)

    print(accumlation)

if __name__ == '__main__':

    boundary_check()
    array_add_types_check()
    wrap_by_jax_array_to_solve_inefficiency()
