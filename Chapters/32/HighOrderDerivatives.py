import jax

def function(x):
    
    # f(x) = 3x³ - 2x² + x - 1
    return 3 * x ** 3 - 2 * x ** 2 + x - 1

def test():
    
    # f'(x) = 9x²- 4x + 1
    dfx_dx = jax.grad(function)
    print("First order derivative = ", dfx_dx(2.))
    
    # f''(x) = 18x- 4
    dfx_dx2 = jax.grad(dfx_dx)
    print("Second order derivative = ", dfx_dx2(2.))
    
if __name__ == "__main__":
    
    test()
