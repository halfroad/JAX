import jax
from jax.experimental import sparse


def sparse_array_transpose():

    array = jax.numpy.array([[0., 1., 0., 2.],
                             [3., 0., 0., 0.],
                             [0., 0., 4., 0.]
                             ])
    # Convert the general array to sparse array
    sparsed_array = sparse.BCOO.fromdense(array)

    dot_array = jax.numpy.array([
        [1.],
        [2.],
        [3]
    ])

    # T is to tranpose the sparse array
    print(sparsed_array.T)

    # @ is new operator to compute the matrix
    dots = sparsed_array.T@dot_array

    print("Dots = ", dots)

    # It has to convert the sparse matrix to general matrix if matrixs to be computed by jax.numpy
    dense = sparsed_array.T.todense()
    dots = jax.numpy.dot(dense, dot_array)

    print("Dots = ", dots)



if __name__ == '__main__':

    sparse_array_transpose()
