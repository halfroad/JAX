import jax
import jax.numpy as jnp

def name_dimensions():

    key = jax.random.PRNGKey(17)

    weights = jax.random.normal(key, shape = (2, 2))
    biases = jax.random.normal(key + 1, shape = (2,))

    parameters = dict(weight = weights, bias = biases)

    print("parameters = ", parameters)

    mapped = jax.tree_util.tree_map(lambda x: x.shape, parameters)

    print("mapped = ", mapped)

def start():

    name_dimensions()

def main():

    start()

if __name__ == "__main__":

    main()
