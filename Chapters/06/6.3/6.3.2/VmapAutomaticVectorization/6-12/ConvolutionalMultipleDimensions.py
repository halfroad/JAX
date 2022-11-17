import jax


def convolve(inputs, weights):

    outputs = []

    for i in range(1, len(inputs) - 1):

        slices = inputs[i - 1: i + 2]
        dotted = jax.numpy.dot(slices, weights)

        outputs.append(dotted)

    return outputs

def start():

    inputs = jax.numpy.arange(5)
    print("inputs = ", inputs)

    weights = jax.numpy.array([2., 3., 4.])
    print("weights = ", weights)

    inputs_ = jax.numpy.stack([inputs, inputs])
    print("inputs_ = ", inputs_)

    weights_ = jax.numpy.stack([weights, weights])
    print("weights_ = ", weights_)

    outputs = convolve(inputs_, weights_)
    print("outputs = ", outputs)

    auto_batch_convolve = jax.vmap(convolve)

    outputs = auto_batch_convolve(inputs_, weights_)
    print("auto_batch_convolve outputs = ", outputs)

    auto_batch_convolve_v2 = jax.vmap(convolve, in_axes = 1, out_axes = 1)

    transpose_inputs = jax.numpy.transpose(inputs_)
    transpose_weights = jax.numpy.transpose(weights_)

    outputs = auto_batch_convolve_v2(transpose_inputs, transpose_weights)

    print("auto_batch_convolve_v2 outputs = ", outputs)

    # Be noted the input dimensions of in_axes
    auto_batch_convolve_v3 = jax.vmap(convolve, in_axes = [0, None])

    outputs = auto_batch_convolve_v3(transpose_inputs, transpose_weights)

    print("auto_batch_convolve_v3 outputs = ", outputs)


if __name__ == '__main__':

    start()

