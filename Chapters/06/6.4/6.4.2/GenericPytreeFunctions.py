import jax

def generic_functions():

    list_of_lists = [[1, 2, 3], [1, 2], [2, 3, 4, 5]]

    mapped = jax.tree_util.tree_map(lambda x: x * 2, list_of_lists)

    print(mapped)

    # The dimensions should be identical
    first_list = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    second_list = [[1, 0, 1], [1, 1, 1], [0, 0, 0]]

    mapped = jax.tree_util.tree_map(lambda x, y: x + y, first_list, second_list)

    print(mapped)

def start():

    generic_functions()

def main():

    start()

if __name__ == "__main__":

    main()
