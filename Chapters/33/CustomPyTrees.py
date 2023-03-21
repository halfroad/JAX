import jax


class CustomTree:
    
    def __init__(self, name: str, age: int, height: int, weight: int):
        
        self.name = name
        
        self.age = age
        self.height = height
        self.weight = weight
        
def flatten(container: CustomTree):
    
    name = container.name
    contents = [container.age, container.height, container.weight]
    
    return contents, name

def unflatten(name: str, contents: list) -> CustomTree:
    
    # contents: Auto unboxing
    container = CustomTree(name, contents)
    
    return container

def generate_leaves():
    
    leaves = jax.tree_util.tree_leaves([
        CustomTree("Name 1", 1, 2, 3),
        CustomTree("Name 2", 4, 5, 6),
        CustomTree("Name 3", 7, 8, 9)])
    
    return leaves

def test():
    
    leaves = generate_leaves()
    
    print("leaves = ", leaves)
    
    print("------------------------------")
    
    jax.tree_util.register_pytree_node(CustomTree, flatten_func = flatten, unflatten_func = unflatten)
    leaves = generate_leaves()
    
    print("leaves = ", leaves)
    
if __name__ == "__main__":
    
    test()