import jax

def transpose(trees):
    
    new_trees = jax.tree_util.tree_map(lambda *xs: list(xs), *trees)
    
    return new_trees

def test():
    
    episodes = [dict(t = 1, obs = 3), dict(t = 2, obs = 4)]
    transposed = transpose(episodes)
    
    print("transposed = ", transposed)
    
if __name__ == "__main__":
    
    test()
    
    