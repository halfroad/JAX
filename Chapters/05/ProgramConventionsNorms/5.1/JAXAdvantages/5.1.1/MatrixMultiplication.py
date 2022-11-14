import jax


def matrix_multiplify():

    prng = jax.random.PRNGKey(15)

    matrix1 = jax.random.normal(prng, shape = [2, 3])
    matrix2 = jax.random.normal(prng, shape = [3, 1])

    print(f"matrix1 = {matrix1},\n matrix2 = {matrix2}")

    multiplified_matrix = jax.numpy.matmul(matrix1, matrix2)

    print(f"multiplified_matrix = {multiplified_matrix}")

    multiplified_matrix = jax.numpy.dot(matrix1, matrix2)

    print(f"multiplified_matrix = {multiplified_matrix}")

def start():

    matrix_multiplify()

if __name__ == "__main__":

    start()
