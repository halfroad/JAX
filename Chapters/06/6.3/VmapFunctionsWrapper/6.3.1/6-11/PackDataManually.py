import jax.numpy


def convolve(inputs, weights):

    outputs = []

    for i in range(1, len(inputs) - 1):

        outputs.append(jax.numpy.dot(inputs[i - 1: i + 2], weights))

    return jax.numpy.array(outputs)

def manually_batched_convolve(inputs, weights):

    outputs = []

    for i in range(inputs.shape[0]):

        outputs.append(convolve(inputs[i], weights[i]))

    return jax.numpy.stack(outputs)

def manually_vectorized_convolve(inputs, weights):

    outputs = []

    for i in range(1, inputs.shape[-1] - 1):

        outputs.append(jax.numpy.sum(inputs[:, i - 1: i + 2] * weights, axis = 1))

    return jax.numpy.stack(outputs, axis = 1)

def manually_vectorized_convolve_v1(inputs, weights):

    outputs = []

    for i in range(1, inputs.shape[-1] - 1):

        outputs.append((inputs[:, i - 1: i + 2] @ weights.T))

    return jax.numpy.stack(outputs, axis = 1)

def start():

    inputs = jax.numpy.arange(5)
    weights = jax.numpy.array([2., 3., 4.])

    outputs = convolve(inputs, weights)

    print(outputs)
    print("-------------------------")

    inputs = jax.numpy.stack([inputs, inputs])
    weights = jax.numpy.stack([weights, weights])

    outputs = manually_batched_convolve(inputs, weights)

    print(outputs)
    print("-------------------------")


    outputs = manually_vectorized_convolve(inputs, weights)

    print(outputs)
    print("-------------------------")

    outputs = manually_vectorized_convolve_v1(inputs, weights)

    print(outputs)

if __name__ == '__main__':

    start()

