import time

import jax.random


def normalize(inputs):

    inputs = inputs - inputs.mean(0)

    return inputs / inputs.std(0)

def start():

    prng = jax.random.PRNGKey(15)
    inputs = jax.random.normal(prng, shape = [1024, 1024])

    begin = time.time()

    normalize(inputs)

    end = time.time()

    print("Time consumed while iterating normalization array: %.2fs" % (end - begin))

    jit_normalize = jax.jit(normalize)

    begin = time.time()

    jit_normalize(inputs)

    end = time.time()

    print("Time consumed while iterating normalization array: %.2fs" % (end - begin))

if __name__ == "__main__":

    start()
