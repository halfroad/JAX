import jax
import jax.experimental.sparse


def create_sparse_array():

    array = jax.numpy.array([
        [0., 1, 0., 2.],
        [3., 0., 0., 0.],
        [0., 0., 4., 0.]
    ])

    sparsed_array = jax.experimental.sparse.BCOO.fromdense(array)

    print("sparsed_array =", sparsed_array)

    return sparsed_array

def transpose(sparsed_array):

    dot_array = jax.numpy.array([
        [1.],
        [2.],
        [2.]
    ])

    print("sparsed_array.T =", sparsed_array.T)
    print("sparsed_array.T@dot_array =", sparsed_array.T@dot_array)
    print("sparsed_array.T.todense() =", sparsed_array.T.todense())

    dotted = jax.numpy.dot(sparsed_array.T.todense(), dot_array)

    print("dotted =", dotted)

def jit_vmap_grad(sparsed_array):

    @jax.jit
    def function(dot_array):

        return (sparsed_array.T @ dot_array).sum()

    dot_array_ = jax.numpy.array([
        [1.],
        [2.],
        [3]
    ])

    grad_function = jax.grad(function)

    print(grad_function(dot_array_))

def sparsify(sparsed_array):

    def function(sparsed_array_, dot_array_):
        return jax.numpy.dot(sparsed_array_.T, dot_array_)

    dot_array = jax.numpy.array([
        [1.],
        [2.],
        [2.]
    ])

    # print(function(sparsed_array, dot_array))

    sparsified_function = jax.experimental.sparse.sparsify(function)

    print(sparsified_function(sparsed_array, dot_array))

def start():

    sparsed_array = create_sparse_array()

    print("---------------------------------")

    transpose(sparsed_array)

    print("---------------------------------")

    jit_vmap_grad(sparsed_array)

    print("---------------------------------")

    sparsify(sparsed_array)

def main():

    start()

if __name__ == "__main__":

    main()
