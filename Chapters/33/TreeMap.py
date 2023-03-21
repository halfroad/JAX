import jax

def test():
    
    _list = [[1, 2, 3], [1, 2], [2, 3, 4, 5]]
    _list = jax.tree_util.tree_map(lambda x: x * 2, _list)
    
    print("_list =", _list)
    
if __name__ == "__main__":
    
    test()
    