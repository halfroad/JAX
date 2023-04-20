import jax

def function(x):
    
    return 2 * x ** 2 + 3 * x

def test():
    
    function_jit = jax.jit(function)
    result = function_jit(10)
    
    print("result = ", result)
    
def main():
    
    test()
    
if __name__ == "__main__":
    
    main()