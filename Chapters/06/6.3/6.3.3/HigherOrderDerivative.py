import jax

def derivative(inputs):

    """

    f(x) = x^3 + 2x^2 - 3x + 1   (orginal y = f(x))

    df(x)/dx = 3x^2 + 4x - 3    (first oder derivative)

    d^2f(x)/dx^2 = 6x + 4       (second order derivative)

    """

    function = lambda x: x ** 3 + 2 * x ** 2 - 3 * x + 1

    first_order_dfdx = jax.grad(function)
    first_order_derivative = first_order_dfdx(inputs)

    second_order_dfdx = jax.grad(first_order_dfdx)
    second_order_derivative = second_order_dfdx(inputs)

    jacfwd_function = jax.jacfwd(function)
    jacrev_function = jax.jacrev(function)

    jacfwd_derivative = jacfwd_function(inputs)
    jacrev_derivative = jacrev_function(inputs)

    return first_order_derivative, second_order_derivative, jacfwd_derivative, jacrev_derivative

def start():

    inputs = 1.0

    first_order_derivative, second_order_derivative, jacfwd_derivative, jacrev_derivative = derivative(inputs)

    print("first_order_derivative = %.1f," % first_order_derivative, f"second_order_derivative = %.1f" % second_order_derivative, f"jacfwd_derivative = %.1f" % jacfwd_derivative, f"jacrev_derivative = %.1f" % jacrev_derivative)

def main():

    start()

if __name__ == "__main__":

    main()
