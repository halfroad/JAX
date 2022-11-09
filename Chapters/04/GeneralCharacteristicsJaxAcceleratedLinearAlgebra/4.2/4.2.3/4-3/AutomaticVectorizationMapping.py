import time

import jax.numpy


def sum_logistic(inputs):

    return jax.numpy.sum(1.0 / 1.0 + jax.numpy.exp(-inputs))

def start():

    begin = time.time()
    inputs = jax.numpy.arange(1024000.)

    derivative_sum_logistic = (jax.grad(sum_logistic))

    end = time.time()

    print("Time consumed while iterating element-wise array: %.2fs" % (end - begin))

    begin = time.time()
    inputs = jax.numpy.arange(1024000.)

    derivative_sum_logistic = jax.vmap(jax.grad(sum_logistic))

    end = time.time()

    print("Time consumed while iterating element-wise array: %.2fs" % (end - begin))

    begin = time.time()

    inputs = jax.numpy.arange(1024000.)

    derivative_sum_logistic = jax.jit(jax.vmap(jax.grad(sum_logistic)))

    end = time.time()

    print("Time consumed while iterating element-wise array: %.2fs" % (end - begin))

if __name__ == "__main__":

    start()
