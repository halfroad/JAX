import jax

def function(x):
    
    if x < 3:
        return 3. * x ** 2
    else:
        return -4 * x
    
def test():
    
    result = function(5)
    
    print(f"function result = ", result)
    print("--------------------------------")
    
    function_jit = jax.grad(function)
    
    result = function_jit(2.0)
    print(f"function_jit(2.0) result = ", result)
    print("--------------------------------")
    
    result = function_jit(5.0)
    print(f"function_jit(5.0) result = ", result)
    print("--------------------------------")
    
    result = function_jit(4)
    print(f"function_jit(4) result = ", result)
    print("--------------------------------")
    
if __name__ == "__main__":
    
    test()