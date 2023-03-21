import jax

def test():
    
    list1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    list2 = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    
    _list = jax.tree_util.tree_map(lambda x, y: x + y, list1, list2)
    
    print("_list = ", _list)
    
if __name__ == "__main__":
    
    test()
    
    
    
    