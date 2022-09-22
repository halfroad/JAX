import jax

def function(x, y):

    return 2 * x ** 3 + y ** 3 + 3 * x ** 2 * y + 4 * x + 5 * y + 6

def mean_squared_error(inputs, genuines):

    weight = .3
    bias = .2

    predictions = weight * inputs + bias
    differences = genuines - predictions
    squared_differences = differences ** 2

    losses = jax.numpy.mean(squared_differences)

    return losses

def start():

    grad_function = jax.grad(function)

    result = grad_function(2., 3.)

    print(result)

    second_grad_function = jax.grad(grad_function, argnums = (0, 1))

    result = second_grad_function(2., 3.)

    print(result)

    grad_function = jax.grad(function, argnums = (0, 1))

    result = grad_function(2., 3.)

    print(result)

    grad_mean_squared_error = jax.grad(mean_squared_error)

    inputs = jax.numpy.linspace(0, 100, 2)
    genuines = inputs * 0.3 + 0.2

    grad_mean_squared_error(inputs, genuines)


def main():

    start()

if __name__ == "__main__":

    main()
