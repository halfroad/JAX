import jax.numpy


def function(matrix1, matrix2):

    print("Running function():")
    print(f"matrix1 = {matrix1}")
    print(f"matrix2 = {matrix2}")

    dotted = jax.numpy.dot(matrix1 + 1, matrix2 + 1)

    print(f"Dotted = {dotted}")

    return dotted

def start():

    prng = jax.random.PRNGKey(15)

    matrix1 = jax.random.normal(prng, shape = [5, 3])
    matrix2 = jax.random.normal(prng, shape = [3, 4])

    """
    
    matrix1 = jax.numpy.arange(1, 16, 1)
    matrix2 = jax.numpy.arange(1, 13, 1)

    matrix1 = matrix1.reshape([5, 3])
    matrix2 = matrix2.reshape([3, 4])
    
    """

    function(matrix1, matrix2)

    jit_function = jax.jit(function)

    jit_function(matrix1, matrix2)

if __name__ == "__main__":

    start()
