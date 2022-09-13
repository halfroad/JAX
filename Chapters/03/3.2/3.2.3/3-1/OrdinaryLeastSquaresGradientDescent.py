import jax.numpy as jnp


def error_function(theta, x, y, m):
    """

    Error Function
    Parameter theta: a [2, 1] matrix, used to compute against input x, and obtain the computed prediction y_predicted. y_predicted is the errors with y.
    
    """

    h_predicted = jnp.dot(x, theta)
    print("h_predicted:", h_predicted)

    transposed = jnp.transpose(h_predicted)
    dotted = jnp.dot(transposed, h_predicted)

    j_theta = (1. / (2 * m)) * dotted

    print("j_theta:", j_theta)

    return j_theta

def gradient_function(theta, x, y, m):

    dotted = jnp.dot(x, theta)
    h_predicted = dotted - y

    print("h_predicted:", h_predicted)

    transposed = jnp.transpose(x)
    print("transposed:", transposed)

    dotted = jnp.dot(transposed, h_predicted)
    print("dotted:", dotted)

    gradient = (1. / m) * dotted

    print("gradient:", gradient)

    return gradient


def gradient_descent(x, y, alpha, m):

    # Shape [2, 1] is the parameter of theta
    theta = jnp.array([1, 1]).reshape(2, 1)

    print("theta:", theta)

    gradient = gradient_function(theta, x, y, m)

    print("gradient:", gradient)

    """
    
    jnp.any([True, False, True]) is true
    jnp.all([True, False, True) is false
    jnp.all([True, True, True) is true
    
    """
    while not jnp.all(jnp.absolute(gradient) <= 1e-5):

        theta = theta - alpha * gradient
        print("theta:", theta)

        gradient = gradient_function(theta, x, y, m)
        print("gradient:", gradient)

    return theta


def start():

    m = 20

    # Generate the dataset x. Here dataset x is a 2 dimensions matrix
    x0 = jnp.ones((m, 1))

    print("x0:", x0)

    x1 = jnp.arange(1, m + 1).reshape(m, 1)

    print("x1:", x1)

    # [20, 2]
    x = jnp.hstack((x0, x1))

    print("x:", x)

    y = jnp.array([3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12, 11, 13, 13, 16, 17, 18, 17, 19, 21]).reshape(m, 1)

    print("y:", y)

    alpha = 0.01

    theta = gradient_descent(x, y, alpha, m)

    print("Optimal:", theta)

    error = error_function(theta, x, y, m)

    print("Error Function:", error)

    error = error[0, 0]

    print("Error Function:", error)


def main():
    start()


if __name__ == "__main__":
    main()

"""

x0: [[1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]]
x1: [[ 1]
 [ 2]
 [ 3]
 [ 4]
 [ 5]
 [ 6]
 [ 7]
 [ 8]
 [ 9]
 [10]
 [11]
 [12]
 [13]
 [14]
 [15]
 [16]
 [17]
 [18]
 [19]
 [20]]
x: [[ 1.  1.]
 [ 1.  2.]
 [ 1.  3.]
 [ 1.  4.]
 [ 1.  5.]
 [ 1.  6.]
 [ 1.  7.]
 [ 1.  8.]
 [ 1.  9.]
 [ 1. 10.]
 [ 1. 11.]
 [ 1. 12.]
 [ 1. 13.]
 [ 1. 14.]
 [ 1. 15.]
 [ 1. 16.]
 [ 1. 17.]
 [ 1. 18.]
 [ 1. 19.]
 [ 1. 20.]]
y: [[ 3]
 [ 4]
 [ 5]
 [ 5]
 [ 2]
 [ 4]
 [ 7]
 [ 8]
 [11]
 [ 8]
 [12]
 [11]
 [13]
 [13]
 [16]
 [17]
 [18]
 [17]
 [19]
 [21]]
 
"""
