import jax

def get_negatives(inputs):
    
    return inputs[inputs < 0]

def test():
    
    prng = jax.random.PRNGKey(15)
    inputs = jax.random.normal(key = prng, shape = (10, 10))
    
    print(get_negatives(inputs))
    
    jit_get_negatives = jax.jit(get_negatives)
    
    print(jit_get_negatives(inputs))
    
if __name__ == "__main__":
    
    test()