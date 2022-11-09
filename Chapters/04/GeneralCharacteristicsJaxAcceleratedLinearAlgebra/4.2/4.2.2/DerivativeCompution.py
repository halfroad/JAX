import jax.numpy


def sum_logistic(inputs):

    return jax.numpy.sum(1.0 / (1.0 + jax.numpy.exp(-inputs)))

def start():

    # Be noted that the data type is float
    inputs = jax.numpy.arange(3.)
    print(f"inputs = {inputs}")

    sums = sum_logistic(inputs)
    print(f"sums = {sums}")

    derivative_sum_logistic = jax.grad(sum_logistic)

    derivatives = derivative_sum_logistic(inputs)
    print(f"derivative = {derivatives}")

if __name__ == "__main__":

    start()

if __name__ == "__main__":

    start()
