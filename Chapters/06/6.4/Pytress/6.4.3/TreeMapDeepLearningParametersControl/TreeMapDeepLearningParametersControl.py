import jax.random


def setup():

    prng = jax.random.PRNGKey(15)

    # Parameters weight and bias are generated respectively. Be noted the name convention in dict
    params = dict(weight = jax.random.normal(key = prng, shape = (2, 2)), bias = jax.random.normal(key = prng + 1, shape = (2,)))
    keys = jax.tree_util.tree_map(lambda x: x.shape, params)

    return prng, params, keys

def main():

    prng, params, keys = setup()

    print("Params = ", params, "\nKeys = ", keys)

if __name__ == '__main__':

    main()


