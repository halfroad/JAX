import jax

def function(x, y):
    
    # f(x, y) = 2x² + 3y³
    return 2 * x ** 2 + 3 * y ** 3

def test():
    
    x = 2.
    y = 3.
    
    # (d(x), d(y)) = (4x, 9y²)
    function_grad = jax.grad(function, argnums = (0, 1))
    dx, dy = function_grad(x, y)
    
    print("dx = ", dx, ", dy = ", dy)
    
if __name__ == "__main__":
    
    test()
    