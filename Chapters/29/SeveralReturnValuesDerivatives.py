import jax

def function(x, y):
    
    # f(x, y) = 2x² * 3y³
    # g(x, y) = 4x³ + 5y⁴
    return 2 * x ** 2 * 3 * y ** 3, 4 * x ** 3 + 5 * y ** 4

def test():
    
    x = 2.
    y = 3.
    
    # Derivatives = [(dx, dy), (dx, dy)] = [(4x * 3y³, 2x² * 9y²), (12x², 20y³)] = [(648, 648), (48, 540)]
    #function_grad = jax.grad(function, argnums = (0, 1), has_aux = True)
    # derivatives = function_grad(x, y)
    
    df1_dx, df2_dx = jax.jacobian(function, argnums = 0)(x, y)
    df1_dy, df2_dy = jax.jacobian(function, argnums = 1)(x, y)
    
    print("derivatives = ", ((df1_dx, df1_dy), (df2_dx, df2_dy)))
    
if __name__ == "__main__":
    
    test()