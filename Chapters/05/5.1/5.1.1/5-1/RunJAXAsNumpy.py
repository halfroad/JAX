import jax.numpy as jnp
import jax.random


def run():

    array = jnp.linspace(0, 9, 10)

    print("array = ", array)

    key = jax.random.PRNGKey(17)

    matrix1 = jax.random.normal(key, shape=[2, 3])

    print("matrix1 = ", matrix1)

    matrix2 = jax.random.normal(key, shape=[3, 1])

    print("matrix2 = ", matrix2)

    multiplication1 = jax.numpy.matmul(matrix1, matrix2)

    print("multiplication1 = ", multiplication1)

    multiplication2 = jax.numpy.dot(matrix1, matrix2)

    print("multiplication2 = ", multiplication2)

    for i in range(len(matrix1)):

        total = .0

        for j in range(len(matrix1[i])):

            x = matrix1[i][j]
            y = matrix2[j][i]

            total += x * y

        print([total])

def jax_array_immutable():

    array = jnp.linspace(0, 9, 10)

    print(type(array))

    # error
    # array[0] = 17

    array_ = array.at[0].set(17)

    print("array_ = ", array_)
    print("array = ", array)

def main():

    run()
    jax_array_immutable()


if __name__ == "__main__":
    main()
