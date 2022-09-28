import jax

def batch_normalize(inputs, gamma = .9, beta = .25, epsilon = 1e-9):

    # Mean of inputs for 1st dimension
    u = inputs.mean(axis = 0)

    # Variances/Deviations
    variances = inputs.var(axis = 0) + epsilon

    # Arithmetic Square Root
    std = jax.numpy.sqrt(variances)

    # Normalization
    normalization = (inputs - u) / std

    return gamma * normalization + beta
