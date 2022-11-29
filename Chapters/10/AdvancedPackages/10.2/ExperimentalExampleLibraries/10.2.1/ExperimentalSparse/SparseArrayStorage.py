import jax
from jax.experimental import sparse


def sparse_general_array():

    array = jax.numpy.array([[0., 1., 0., 2.],
                             [3., 0., 0., 0.],
                             [0., 0., 4., 0.]
                             ])
    # Convert the general array to sparse array
    sparsed_array = sparse.BCOO.fromdense(array)

    print("General array = ", array, "\nsparsed_array = ", sparsed_array)
    print("sparsed_array.data = ", sparsed_array.data)

    dense = sparsed_array.todense()
    print("dense = ", dense)

    print("sparsed_array.indices = ", sparsed_array.indices)

    for i, j in zip(sparsed_array.indices[0], sparsed_array.indices[1]):

        print(array[i, j])

    # Number of dimensions of original matrix
    print("sparsed_array.ndim = ", sparsed_array.ndim)

    # Size of dimensions of original matrix
    print("sparsed_array.shape = ", sparsed_array.shape)

    # Data type of original matrix
    print("sparsed_array.dtype = ", sparsed_array.dtype)

    # Number of those elements who is 0 on original matrix
    print("sparsed_array.nse = ", sparsed_array.nse)


if __name__ == '__main__':

    sparse_general_array()
