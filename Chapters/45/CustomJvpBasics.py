import jax

@jax.custom_jvp
def function(x):
    
    return 2 * x ** 2 + 3 * x

def function_jvp(primals, tangents):
    
    x, = primals
    x_dot, = tangents
    
    result = function(x)
    result_dot = 4 * x + 3 * x_dot
    
    return result, result_dot
    
def register():
    
    function.defjvp(function_jvp)
    
def main():
    
    register()
    
    a = 5.
    
    result = function(a)
    
    print("result = ", result)
    print("-------------------------")
    
    function_grad = jax.grad(function)
    
    result = function_grad(a)
    
    print("result = ", result)
    print("-------------------------")
    
    b = 6.
    
    result, result_dot = jax.jvp(function, (a,), (b,))
    
    print(f"result = {result}, result_dot = {result_dot}")
    
if __name__ == "__main__":
    
    main()