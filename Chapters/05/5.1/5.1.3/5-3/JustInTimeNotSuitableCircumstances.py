import jax
import jax.random
import time

def normalize(x):

    x = x - x.mean(0)

    return x / x.std(0)

def get_negatives(x):

    return x[x < 0]

def start():

    key = jax.random.PRNGKey(17)

    x = jax.random.normal(key, shape = [1024, 1024])

    begin = time.time()

    normalize(x)

    end = time.time()

    print("Duration {:.2f}s consumed when iterating {} times".format(end - begin, len(x)))

    jit_normalize = jax.jit(normalize)

    begin = time.time()

    jit_normalize(x)

    end = time.time()

    print("Duration {:.2f}s consumed when iterating {} times".format(end - begin, len(x)))

    x = jax.random.normal(key, shape = [10, 10])

    negatives = get_negatives(x)

    print(negatives.shape)

    jit_get_negatives = jax.jit(get_negatives)

    negatives = jit_get_negatives(x)

    print(negatives.shape)

def main():

    start()


if __name__ == "__main__":

    main()
