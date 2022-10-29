import jax.random

"""

Paragraph 2.1.2, Implementation of Fully-Connected Layer
Program 2-1
Page 18

"""

def Dense(shape = [2, 1]):

    # 0 is the random send, can be any number
    key = jax.random.PRNGKey(0)

    weights = jax.random.normal(key, shape = shape)
    biases = jax.random.normal(key, shape = (shape[-1],))

    parameters = [weights, biases]

    print(f"weights = {weights}")
    print(f"biases = {biases}")

    # apply_fun is an internal feature of python, which is intrinsic fucntion
    def apply_func(inputs):

        weight, bias = parameters

        return jax.numpy.dot(inputs, weight) + bias

    return apply_func

def start():

    dense = Dense()

    matrix = jax.numpy.array([[1.7, 1.7],
                              [2.14, 2.14]
                              ])
    new_materix = dense(matrix)

    print(f"new_materix = {new_materix}")

if __name__ == "__main__":

    start()
