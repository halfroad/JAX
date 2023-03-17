import jax

def function(x):
    
    # f(x) = x³ + x²
    return x ** 3 + x ** 2

def test():
    
    # f'(x) = 3x² + 2x
    function_value_and_grad = jax.value_and_grad(function)
    (value, gradient) = function_value_and_grad(2.)
    
    print("Function value = ", value, "gradient = ", gradient)
    
if __name__ == "__main__":
    
    test()
    
    