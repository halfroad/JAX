import time
import jax


def sum_logistic(inputs):

    return jax.numpy.sum(1.0 / (1.0 + jax.numpy.exp(-inputs)))

@jax.jit
def jit_sum_logstic(inputs):

    return jax.numpy.sum(1.0 / (1.0 + jax.numpy.exp(-inputs)))

def start():

    begin = time.time()
    inputs = jax.numpy.arange(1024.)

    derivative_jit_sum_logstic = jax.grad(jit_sum_logstic)
    print(f"derivative_jit_sum_logstic(inputs) = {derivative_jit_sum_logstic(inputs)}")

    end = time.time()

    print("Time consumed while iterating element-wise array: %.2f" % (end - begin))

    begin = time.time()
    jit_sum_logistic = jax.jit(sum_logistic)

    derivative_sum_logstic = jax.grad(sum_logistic)

    print(f"derivative_sum_logstic(inputs) = {derivative_sum_logstic(inputs)}")

    end = time.time()

    print("Time consumed while iterating element-wise array: %.2f" % (end - begin))

    begin = time.time()

    derivative_sum_logstic = jax.grad(sum_logistic)
    jit_derivative_sum_logstic = jax.jit(derivative_sum_logstic)

    print(f"jit_derivative_sum_logstic(inputs) = {jit_derivative_sum_logstic(inputs)}")

    end = time.time()

    print("Time consumed while iterating element-wise array: %.2f" % (end - begin))


if __name__ == "__main__":

    start()
