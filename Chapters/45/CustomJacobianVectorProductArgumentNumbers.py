import jax

def function(a, b):
    
    return a ** 3 + b ** 2

@jax.custom_jvp
def function1(a, b):
    
    return a ** 3 + b ** 2

@function1.defjvp
def function1_jvp(primals, tangents):
    
    a, b = primals
    a_dot, b_dot = tangents
    
    primal_output = function1(a, b)
    tangent_output = a_dot + b_dot
    
    return primal_output, tangent_output

def test():
    
    a = 4.
    b = 6.
    
    function_grad = jax.grad(function, argnums = [0, 1])
    result = function_grad(a, b)
     
    print("Derivative of orginal function =", result)
    print("--------------------------")
    
    function1_grad = jax.grad(function1, argnums = [0, 1])
    result = function1_grad(a, b)
     
    print("Derivative of customized JVP function =", result)
    
def main():
    
    test()
    
if __name__ == "__main__":
    
    main()
