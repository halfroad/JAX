import jax

def function(x):
    
    result = jax.lax.cond(x > 0, lambda x: x, lambda x: x + 1, x)
    
    return result

def test():
    
    function_jit = jax.jit(function)
    result = function_jit(10.)
    
    print("result = ", result)
    
    function_jaxpr = jax.make_jaxpr(function)
    result = function_jaxpr(10.)
    
    print("function_jaxpr = ", result)

if __name__ == "__main__":
    
    test()