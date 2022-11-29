import jax
from jax.experimental import sparse


def function(sparsed_array, dot_array):

    dots = jax.numpy.dot(sparsed_array.T, dot_array)

    return dots

def run():

    dot_array = jax.numpy.array([
        [1.],
        [2.],
        [2.]
    ])

    array = jax.numpy.array([[0., 1., 0., 2.],
                             [3., 0., 0., 0.],
                             [0., 0., 4., 0.]
                             ])
    # Convert the general array to sparse array
    sparsed_array = sparse.BCOO.fromdense(array)

    # TypeError: dot requires ndarray or scalar arguments, got <class 'jax.experimental.sparse.bcoo.BCOO'> at position 0.
    # function(sparsed_array = sparsed_array, dot_array = dot_array)

    function_sparsified = sparse.sparsify(function)

    sparsified_array = function_sparsified(sparsed_array, dot_array)

    print(sparsified_array)

if __name__ == '__main__':

    run()
