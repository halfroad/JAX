import time

import jax
import jax.numpy as jnp

from jax import random


def selu(x, alpha = 1.67, lmbda = 10.5):

    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

@jax.jit
def selu1(x, alpha = 1.67, lmbda = 1.05):

    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

def start():

    prng = random.PRNGKey(17)
    x = random.normal(prng, (1000000,))

    begin = time.time()

    selu(x)

    end = time.time()

    print("{:.2f}s is consumed when iterating x {} times".format(end - begin, len(x)))

    self_jit = jax.jit(selu)

    begin = time.time()

    selu(x)

    end = time.time()

    print("{:.2f}s is consumed when iterating x {} times".format(end - begin, len(x)))

    begin = time.time()

    selu1(x)

    end = time.time()

    print("{:.2f}s is consumed when iterating x {} times".format(end - begin, len(x)))

def main():

    start()


if __name__ == "__main__":

    main()
