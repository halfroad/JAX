import jax.tree_util


class CustomPytree:

    def __init__(self, name: str, a: int, b: int, c: int):

        self.name = name
        self.a = a
        self.b = b
        self.c = c

def flatten_custom_pytree(tree: CustomPytree):

    flat_contents = [tree.a, tree.b, tree.c]
    auxillary = tree.name

    return flat_contents, auxillary

def unflatten_custom_pytree(auxillary:str, flatten_contents: list) -> CustomPytree:

    # Auto Unboxing
    return CustomPytree(auxillary, flatten_contents)

def start():

    leaves = jax.tree_util.tree_leaves([
        CustomPytree("Xiaohua", 1, 2, 3),
        CustomPytree("Xiaoming", 3, 2, 1),
        CustomPytree("Xiaohui", 2, 3, 1)
    ])

    print(leaves)

    jax.tree_util.register_pytree_node(CustomPytree, flatten_custom_pytree, unflatten_custom_pytree)

    leaves = jax.tree_util.tree_leaves([
        CustomPytree("Xiaohua", 1, 2, 3),
        CustomPytree("Xiaoming", 3, 2, 1),
        CustomPytree("Xiaohui", 2, 3, 1)
    ])

    print(leaves)

def main():

    start()

if __name__ == "__main__":

    main()
