import jax.numpy


def pytree_exemplify():

    """

    Below definition could be recognized as a struct of pytree, but not limited to the following sample

    """

    trees = [
        1,      # A sole object, constant 1 also could be recognized as a pytree
        "a",    # A sole object, character "a" also could be recognized as a pytree
        [1, 'a', object()],
        (1, (2, 3), ()),
        [1, {"k1": 2, "k2": (3, 4)}, 5],
        {"a": 2, "b": (2, 3)},
        jax.numpy.array([1, 2, 3])
    ]

    for tree in trees:

        leaves = jax.tree_util.tree_leaves(tree)

        print(f"{tree} has {len(leaves)} leaves: {leaves}")

def start():

    pytree_exemplify()

if __name__ == '__main__':

    start()
