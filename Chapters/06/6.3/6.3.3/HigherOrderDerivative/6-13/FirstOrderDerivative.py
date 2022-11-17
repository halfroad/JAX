import jax


def function(x):

    return x ** 3 + 2 * x ** 2 - 3 * x + 1

def compute_derivative(x):

    dfdx = jax.grad(function)
    derivative = dfdx(x)

    return derivative

def compute_secondary_derivative(x):

    dfdx = jax.grad(function)
    dfdfdxdx = jax.grad(dfdx)

    derivative = dfdfdxdx(x)

    return derivative

def start():

    x = 1.

    derivative = compute_derivative(x)
    print("First Order Derivative = ", derivative)

    derivative = compute_secondary_derivative(x)
    print("Secondary Order Derivative = ", derivative)

if __name__ == '__main__':

    start()
