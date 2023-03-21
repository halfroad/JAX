import jax

def function(x):
    
    if x > 5:
        
        return x
    
    else:
        
        return 2 * x

def test():
    
    # function_jit = jax.jit(function)
    # result = function_jit(10)
    function_grad = jax.grad(function)
    result = function_grad(3.)
    
    print("result = ", result)
    
if __name__ == "__main__":
    
    test()