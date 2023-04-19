import jax

# Attribute that the derivative of this funciton will be customized
@jax.custom_jvp
def function (a, b):
    
    return a ** 3 + b ** 2

@function.defjvp
def function_defjvp (primals, tangents):
    
    print("primals = ", primals)
    print("tangents = ", tangents)
    
    a, b = primals
    a_dot, b_dot = tangents
    
    primal_output = function(a, b)
    tangent_output = 3 * a_dot ** 2 + 0
    
    return primal_output, tangent_output

def test():
    
    a = 4.
    b = 6.
    
    result, result_dot = jax.jvp(function, (a, b), (a, b))
    
    print(f"result = {result}, result_dot = {result_dot}")
    
def main():
    
    test()
    
if __name__ == "__main__":
    
    main()
    
    