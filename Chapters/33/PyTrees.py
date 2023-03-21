import jax

def test():
    
    trees = [1,
             "a",
             [1, 'a', object()],
             [1, (2, 3), ()],
             [1, {"k1": 2, "k2": (3, 4)}, 5],
              {"a": 2, "b": (2, 3)},
              jax.numpy.array([1, 2, 3]),
             ]
    for tree in trees:
        
        leaves = jax.tree_util.tree_leaves(tree)
        
        print(f"{tree} has {len(leaves)} leaves: {leaves}")
        
if __name__ == "__main__":
    
    test()