import jax

@jax.custom_vjp
def function(a, b):
    
    return 2 ** a ** 2 + b ** 2

def function_forward(a, b):
    
    result = function(a, b)
    
    return result, (a, b)

def function_backward(result, gradient):
    
    b, a = result
    
    return (b, a)

def register():
    
    function.defvjp(fwd = function_forward, bwd = function_backward)
    
def test():
    
    register()
    
    a = 4.
    b = 6.
    
    grad_function = jax.grad(fun = function)
    result = grad_function(a, b)
    
    print("result = ", result)
    print("-----------------------")
    
    grad_function = jax.grad(function, argnums = [0, 1])
    result = grad_function(a, b)
    
    print("result = ", result)
    print("-----------------------")

def main():
    
    test()
    
if __name__ == "__main__":
    
    main()