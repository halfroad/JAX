import jax

def function(x):
    # f(x) = x³ + x²
    return x ** 3 + x ** 2

def test():
    # f'(x) = 3x² + 2x
    function_grad = jax.grad(function, allow_int = True)
    
    result = function_grad(2.)
    print("result = ", result)
    
    result = function_grad(3.)
    print("result = ", result)
    
if __name__ == "__main__":
    
    test()
