import jax

def name_dimensions():

    prng = jax.random.PRNGKey(15)
    
    # Parameter weights and biases are generated respectively, note the name in the dict
    params = dict(weights = jax.random.normal(key = prng, shape = (2, 2)), biases = jax.random.normal(key = prng + 1, shape = (2, )))
    print("params = ", params)
    
    print("--------------------------------------")
    
    params = jax.tree_util.tree_map(lambda x: x.shape, params)
    print("params = ", params)
        
def test():

    name_dimensions()
    
if __name__ == "__main__":
    
    test()
