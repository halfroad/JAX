import jax

def add(x):
    
    return x + 1

def substract(x):
    
    return x - 1

def test():
    
    operand = jax.numpy.array([0.])
    
    x = 10.
    
    result = jax.lax.cond(x > 5, add, substract, operand)
    print("Result = ", result)
    print("---------------------------------")
    
    result = jax.lax.cond(x < 5, add, substract, operand)
    print("Result = ", result)
    print("---------------------------------")
    
if __name__ == "__main__":
    
    test()