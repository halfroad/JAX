import jax
import jax.numpy as jnp

def exemplify():

    trees = [
        1,          # A sole object, constant 1 also can be recognized as a pytree
        "a",        # A sole object, string "a" als0 can be recongized as a pytree
        [1, 'a', object()],
        (1, (2, 3), ()),
        [1, {"k1": 2, "k2": (3, 4)}, 5],
        {"a": 2, "b": (2, 3)},
        jnp.array([1, 2, 3]),
    ]

    for tree in trees:

        leaves = jax.tree_util.tree_leaves(tree)

        print(f"{tree} has {len(leaves)}, leaves: {leaves}")

def start():

    exemplify()

def main():

    start()

if __name__ == "__main__":

    main()
