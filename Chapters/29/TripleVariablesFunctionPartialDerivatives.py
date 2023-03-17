import jax

def function(x, y, z):
    
    # f(x, y, z) = 2x² + 3y³ + 4z⁴    
    return 2 * x ** 2 + 3 * y ** 3 + 4 * z ** 4

def test():
    
    x = 2.
    y = 3.
    z = 4.
    
    # Partial Derivates = (dx, dy, dz) = (4x, 9y², 16z³)
    function_grad = jax.grad(function, argnums = (0, 1, 2))
    dx, dy, dz = function_grad(x, y, z)
    
    print("dx = ", dx, " dy = ", dy, " dz = ", dz)
    
if __name__ == "__main__":
    
    test()
    
    
