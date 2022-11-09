"""

Paragraph 4.2.1
Page 68
Accelerate the program by jax.jit

"""
import time
import jax


def selu(inputs, alpha = 1.67, theta = 1.05):

    return theta * jax.numpy.where(inputs > 0, inputs, alpha * (jax.numpy.exp(inputs) - 1))

@jax.jit
def jit_selu(inputs, alpha = 1.67, theta = 1.05):

    return theta * jax.numpy.where(inputs > 0, inputs, alpha * (jax.numpy.exp(inputs) - 1))

def start():

    prng = jax.random.PRNGKey(10)
    begin = time.time()

    inputs = jax.random.normal(prng, shape = (10000,))

    selu(inputs)

    end = time.time()

    print("Time consumed while iterating element-wise array: %.2fs" % (end - begin))

    selu_ = jax.jit(selu)
    begin = time.time()

    selu_(inputs)

    end = time.time()

    print("Time consumed while iterating element-wise array: %.2fs" % (end - begin))

    begin = time.time()

    jit_selu(inputs)

    end = time.time()

    print("Time consuming while iterating element-wise array: %.2fs" % (end - begin))

def main():

    start()

if __name__ == "__main__":

    main()
