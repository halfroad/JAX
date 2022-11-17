import jax


def pytree_manipulate():

    list_of_lists = [[1, 2, 3], [1, 2], [2, 3, 4, 5]]
    iterations = jax.tree_util.tree_map(lambda x: x * 2, list_of_lists)

    print(iterations)

if __name__ == '__main__':

    pytree_manipulate()
