import jax.tree_util


def tree_transpose(list_of_trees):

    # Be noted that here the asterisk * is used to dump the keys in dictionary
    iterations = jax.tree_util.tree_map(lambda *xs: list(xs), *list_of_trees)

    return iterations

def start():

    episode_steps = [dict(t = 1, obs = 2), dict(t = 2, obs = 4)]
    iterations = tree_transpose(episode_steps)

    print(iterations)

if __name__ == '__main__':

    start()
