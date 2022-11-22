import jax


def batch_normalize(inputs, gamma = 0.9, beta = 0.25, epsilon = 1e-9):

    u = jax.numpy.mean(inputs)
    standard_variance = jax.numpy.sqrt(inputs.var(axis = 0) + epsilon)

    # Normalization
    y = (inputs - u) / standard_variance

    scale_shift = y * gamma + beta

    return scale_shift
