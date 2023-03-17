import jax

def add(i, x):
    
    return i + 2., x + 5.

def test():
    
    array = jax.numpy.arange(1, 5)
    result = jax.lax.scan(add, 0, array)
    
    print("array =", array, "result =", result)
    
if __name__ == "__main__":
    
    test()
    
    