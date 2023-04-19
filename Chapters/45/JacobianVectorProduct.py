import jax

def function(a, b):
    
    return a ** 3 + b ** 2

def test():
    
    a = 4.
    b = 6.
    
    result = function(a, b)
    
    print("result = ", result)
    print("-----------------------")
    
    function_grad = jax.grad(function)
    result = function_grad(a, b)
    
    print("result = ", result)
    print("-----------------------")
    
    function_grad = jax.grad(function, argnums = [0, 1])
    result = function_grad(a, b)
    
    print("result = ", result)

def main():
    
    test()
    
if __name__ == "__main__":
    
    main()
    