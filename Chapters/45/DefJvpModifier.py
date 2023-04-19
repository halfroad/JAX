import jax

@jax.custom_jvp
def function(x, y):
    
    return x ** 3 + y ** 2

@function.defjvp
def function_jvp(primals, tangents):
    
    x, y = primals
    x_dot, y_dot = tangents
    
    primal_output = function(x, y)
    tangent_output = x * y_dot + y * x_dot
    
    return primal_output, tangent_output

def main():
    
    a = 5.
    b = 6.
    
    function_grad = jax.grad(function)
    result = function_grad(a, b)
    
    print("result = ", result)
    print("--------------------------")
    
    result, result_dot = jax.jvp(function, (a, b), (a, b))
    
    print(f"result = {result}, result_dot = {result_dot}")
    print("--------------------------")
    
    function_jvp_lambda = lambda primals, tangents: (primals[0], tangents[0])
    
    function.defjvp(function_jvp_lambda)
    function_grad = jax.grad(function)
    result = function_grad(a, b)
    
    print(f"result = {result}")
    print("--------------------------")
    
if __name__ == "__main__":
    
    main()