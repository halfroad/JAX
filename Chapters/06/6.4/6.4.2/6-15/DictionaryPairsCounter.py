import jax

def tree_transpose(list_of_trees):

    return jax.tree_util.tree_map(lambda *xs: list(xs), *list_of_trees)

def start():

    # Be noted the * indicator

    episode_steps = [dict(t = 1, obs = 3), dict(t = 2, obs = 4)]

    transposed = tree_transpose(episode_steps)

    print(transposed)

def main():

    start()

if __name__ == "__main__":

    main()
