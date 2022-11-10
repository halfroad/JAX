import jax.random


def get_negatives(inputs):

    # the dimension passed by parameter is not explicit
    return inputs[inputs < 0]

def start():

    prng = jax.random.PRNGKey(15)
    inputs = jax.random.normal(prng, shape = [10, 10])

    negatives = get_negatives(inputs)

    print(f"negatives.shape = {negatives.shape}")

    jit_get_negatives = jax.jit(get_negatives)

    negatives = jit_get_negatives(inputs)

    print(f"negatives.shape = {negatives.shape}")

if __name__ == "__main__":

    start()
