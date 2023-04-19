import jax

@jax.custom_jvp

def function(x):
    
    return 2 * x + 3

@function.defjvp
def function_jvp(primals, tangents):
    
    x, = primals
    x_dot, = tangents
    
    if x >= 0:
        return function(x), x * x_dot
    else:
        return function(x), -x * x_dot
    
def main():
    
    grad_function = jax.grad(function)
    
    result = grad_function(1.0)
    
    print("result = ", result)
    print("------------------------")
    
    result = grad_function(-1.0)
    print("result = ", result)
    
if __name__ == "__main__":
    
    main()
    
    