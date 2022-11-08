import jax.numpy


def setup():

    total = 20

    # Generate the dataset, the dataset x here is a 2 dimensions matrix
    x0 = jax.numpy.ones((total, 1))
    x1 = jax.numpy.arange(1, total + 1).reshape(total, 1)

    # [20, 2]
    inputs = jax.numpy.hstack((x0, x1))
    genuines = jax.numpy.array([3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
                                11, 13, 13, 16, 17, 18, 17, 19, 21]).reshape(total, 1)

    alpha = 0.01

    return total, inputs, genuines, alpha

def error_function(theta, inputs, total):

    """

    The theta is a matrix with the size [2, 1], which is used to compute against inputs and the computed h_predict as the outcome.
    The y_pred is the error computed with y

    total is m

    """
    prediction = jax.numpy.dot(inputs, theta)
    transposed = jax.numpy.transpose(prediction)

    theta = (1. / 2 * total) * jax.numpy.dot(transposed, prediction)

    return theta

def gradient_function(theta, inputs, genuines, total):

    prediction = jax.numpy.dot(inputs, theta) - genuines
    transposed = jax.numpy.transpose(inputs)

    return (1. / total) * jax.numpy.dot(transposed, prediction)

def gradient_descent(inputs, genuines, alpha, total):

    theta = jax.numpy.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, inputs, genuines, total)

    while not jax.numpy.all(jax.numpy.absolute(gradient) <= 1e-5):

        theta = theta - alpha * gradient
        gradient = gradient_function(theta, inputs, genuines, total)

    return theta

def start():

    total, inputs, genuines, alpha = setup()
    theta = gradient_descent(inputs, genuines, alpha, total)

    print("Optimal:", theta)

    error = error_function(theta, inputs, genuines)

    print("Error Function:", error)

if __name__ == "__main__":

    counter = 0

    for i in range(1, 101):

        counter += i

    print(counter)

    # start()
