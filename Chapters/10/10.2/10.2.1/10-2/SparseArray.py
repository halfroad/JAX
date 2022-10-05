import jax
from jax.experimental import sparse


def start():

    array = jax.numpy.array([
        [0., 1., 0., 2.],
        [3., 0., 0., 0.],
        [0., 0., 4., 0.]
    ])

    sparsed_array = sparse.BCOO.fromdense(array)

    print("sparsed_array =", sparsed_array)
    print("sparsed_array.data =", sparsed_array.data)
    print("sparsed_array.todense() =", sparsed_array.todense())
    print("sparsed_array.indices =", sparsed_array.indices)

    for i, j in zip(sparsed_array.indices[0], sparsed_array.indices[1]):

        print(f"array[{i}, {j}] =", array[i, j])

    print("sparsed_array.ndim =", sparsed_array.ndim)
    print("sparsed_array.shape =", sparsed_array.shape)
    print("sparsed_array.dtype =", sparsed_array.dtype)
    print("sparsed_array.nse =", sparsed_array.nse)

def main():

    start()

if __name__ == "__main__":

    main()
