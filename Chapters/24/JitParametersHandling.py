import jax

def function(x, y):
    
    print("Running fucntion(x, y)")
    
    print(f"x = {x}")
    print(f"y = {y}")
    
    result = jax.numpy.dot(x + 1, y + 1)
    
    print(f"result = {result}")
    
    return result

def test():
    
    prng = jax.random.PRNGKey(15)
    
    x = jax.random.normal(prng, shape = [5, 3])
    y = jax.random.normal(prng, shape = [3, 4])
    
    function(x, y)
    
    print("-----------------")
    
    function_jit = jax.jit(function)
    
    function_jit(x, y)
    
if __name__ == "__main__":
    
    test()