import jax

def function(x, y):
    
    # f(x, y) = 2x² + 3y³
    return 2 * x ** 2 + 3 * y ** 3

def test():
    
    x = 2.
    y = 3.
    
    # f(x, y) = 2x² + 3y³
    # Partial Derivatives
    # d(x) = 4x + 0 = 4x
    # d(y) = 0 + 9y² = 9y²
    function_grad = jax.grad(function, argnums = 1)
    value = function_grad(x, y)
    
    print("Derivative value = ", value)
    
if __name__ == "__main__":
    
    test()
    
    