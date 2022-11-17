import jax.tree_util


def multimap_manipulate():

    first_list = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

    # The dimensions should be identical
    second_list = [[1, 0, 1], [1, 1, 1], [0, 0, 0]]

    iterations = jax.tree_util.tree_map(lambda x, y: x + y, first_list, second_list)

    print(iterations)

if __name__ == '__main__':

    multimap_manipulate()
