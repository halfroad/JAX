import jax

def function(x):
    
    if x < 3:
        return 3. * x ** 2
    else:
        return -4 * x
    
def construct_array_by(length, value):
    
    return jax.numpy.ones(shape = (length,)) * value
    
def test():
    
    '''

    result = function(5)
    
    print(f"function result = ", result)
    print("--------------------------------")
    
    function_grad = jax.grad(function)
    
    result = function_grad(2.0)
    print(f"function_grad(2.0) result = ", result)
    print("--------------------------------")
    
    result = function_grad(5.0)
    print(f"function_grad(5.0) result = ", result)
    print("--------------------------------")
    
    
    result = function_grad(4)
    print(f"function_grad(4) result = ", result)
    print("--------------------------------")
    
    function_jit = jax.jit(function, static_argnums = (0,))
    result = function_jit(2.)
    print(f"function_jit(2.) result = ", result)
    print("--------------------------------")
    
     '''
    
    result = construct_array_by(5, 4)
    print(f"construct_array_by(5, 4) result = ", result)
    print("--------------------------------")
    
    construct_array_by_jit = jax.jit(construct_array_by, static_argnums = (0,))
    result = construct_array_by_jit(5, 4)
    print(f"construct_array_by_jit(5, 4) result = ", result)
    print("--------------------------------")
    
    construct_array_by_jit = jax.jit(construct_array_by, static_argnames = ['length'])
    result = construct_array_by_jit(5, 4)
    print(f"construct_array_by_jit(5, 4) result = ", result)
    print("--------------------------------")
    
if __name__ == "__main__":
    
    test()