import jax

@jax.custom_vjp
def function(x):
    
    return 2 * x ** 2 + 3 * x

def function_forward(x):
    
    return function(x), 4 * x + 3

def function_backward(dot_x, y_bar):
    
    return (dot_x * y_bar,)

def test():
    
    function.defvjp(function_forward, function_backward)
    
    function_grad = jax.grad(function)
    
    result = function_grad(3.0)
    
    print("result = ", result)
    
def main():
    
    test()
    
if __name__ == "__main__":
    
    main()
