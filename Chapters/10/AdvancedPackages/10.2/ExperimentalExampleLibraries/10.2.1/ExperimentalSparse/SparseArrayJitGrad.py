import jax
from jax.experimental import sparse


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

    print(jit_vmap_grad(sparsed_array))

if __name__ == '__main__':

    run()
