import jax


def compute_one_dimensional():

    key = jax.random.PRNGKey(15)

    inputs = jax.numpy.linspace(0, 9, 10)
    print("Inputs = ", inputs)

    kernel = jax.numpy.ones(3) / 10
    print("Kernel = ", kernel)

    outputs = jax.numpy.convolve(inputs, kernel)

    print("Outputs with default model = ", outputs)

    outputs = jax.numpy.convolve(inputs, kernel, mode = "full")

    print("Outputs with full model = ", outputs)

    outputs = jax.numpy.convolve(inputs, kernel, mode = "same")

    print("Outputs with same model = ", outputs)

    outputs = jax.numpy.convolve(inputs, kernel, mode = "valid")

    print("Outputs with valid model = ", outputs)

if __name__ == '__main__':

    compute_one_dimensional()
