import jax

def function(x):
    
    if x > 0:
        
        return x
    
    else:
        
        return 2 * x
    
def test():
    
    function_jit = jax.jit(function, static_argnums = (0,))
    result = function_jit(10.)
    
    print("result = ", result)
    
    function_jaxpr = jax.make_jaxpr(function, static_argnums = (0,))
    result = function_jaxpr(10.)
    
    print("result = ", result)
    
if __name__ == "__main__":
    
    test()
                           