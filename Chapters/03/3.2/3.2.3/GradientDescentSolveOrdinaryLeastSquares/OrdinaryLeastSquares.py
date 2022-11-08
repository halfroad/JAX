import jax


def error_function(theta, x, y, m):

    h_predict = jax.numpy.dot(x, theta)
    transposed = jax.numpy.transpose(h_predict)
    j_theta = 1./(2 * m) * jax.numpy.dot(transposed, h_predict)

    return j_theta

def gradient_function(theta, x, y, m):

    h_predict = jax.numpy.dot(x, theta) - y
    transposed = jax.numpy.transpose(x)

    return 1./ m * jax.numpy.dot(transposed, h_predict)

def gradient_descendt(x, y, alpha, m):

    # Here the theta is parameter
    theta = jax.numpy.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, x, y, m)

    while not jax.numpy.all(jax.numpy.absolute(gradient) <= 1e-5):

        theta = theta - alpha * gradient
        gradient = gradient_function(theta, x, y, m)

    return theta
