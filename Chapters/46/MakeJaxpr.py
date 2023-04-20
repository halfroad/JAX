import jax

def function(x):
    
    return 2 * x ** 2 + 3 * x

def print_jaxpr(closed_expr):
    
    jaxpr = closed_expr.jaxpr
    
    print("invars: ", jaxpr.invars)
    print("outvars: ", jaxpr.outvars)
    print("constvars: ", jaxpr.constvars)
    
    for equation in jaxpr.eqns:
        print("Equation: ", equation.invars, equation.primitive, equation.outvars, equation.params)
        
    print("jaxpr: ", jaxpr)
    
def print_literals():
    
    function_jaxpr = jax.make_jaxpr(function)
    closed_jaxpr = function_jaxpr(2.0)
    
    print(closed_jaxpr)
    print("-----------------------------------------")
    
    print(closed_jaxpr.literals)

def test():
    
    expr = jax.make_jaxpr(function)
    result = expr(2.0)
    
    print(result)
    print("-----------------------------------------")
    
    print_jaxpr(result)
    
    print("-----------------------------------------")
    
    print_literals()
    
def main():
    
    test()
    
if __name__ == "__main__":
    
    main()