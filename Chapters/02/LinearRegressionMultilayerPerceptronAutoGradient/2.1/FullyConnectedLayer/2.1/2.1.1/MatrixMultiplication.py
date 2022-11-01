import jax.numpy

"""

Paragraph 2.1.1, Fully-Connected Layer: Hidden Layer of Multilayer Perceptron

Page 17

"""

def multiplify():

    """

    [
        1.7    1.7
        2.14    2.14
    ]
    @
    [
        1
        2
    ]
    +
    0.99

    """

    matrix = jax.numpy.array([

        [1.7, 1.7],
        [2.14, 2.14]
    ])
    weight = jax.numpy.array([[1], [2]])
    bias = .99

    multiplified = jax.numpy.matmul(matrix, weight) + bias

    print(f"Multiplified = {multiplified}, Multiplified.shape = {multiplified.shape}, Multiplified.ndim = {multiplified.ndim}")

    dotted = jax.numpy.dot(matrix, weight) + bias

    print(f"dotted = {dotted}, dotted.shape = {dotted.shape}, dotted.ndim = {dotted.ndim}")

if __name__ == "__main__":

    multiplify()
