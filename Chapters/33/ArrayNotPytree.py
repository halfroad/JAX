import jax

if __name__ == "__main__":
    
    trees = [1, 2, 3]
        
    
    print(jax.tree_util.tree_map(lambda x: x + 1, trees))
             
             