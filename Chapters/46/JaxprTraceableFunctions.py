import jax

def inverse_iterate_jaxpr(inverse_registry, jaxpr, consts, *args):
    
    configurations = {}
    
    def read(var):
        
        if type(var) is jax.core.Literal:
            
            return var.val
        
        return configurations[var]
    
    def write(var, value):
        
        configurations[var] = value
    
    jax.util.safe_map(write, jaxpr.outvars, args)
    jax.util.safe_map(write, jaxpr.constvars, consts)
    
    # Backwards iteration
    for equation in jaxpr.eqns[:: -1]:
        
        in_values = jax.util.safe_map(read, equation.outvars)
        
        if equation.primitive not in inverse_registry:
            
            raise NotImplementedError("{} does not registered inverse.".format(equation.primitive))
        
        out_values = inverse_registry[equation.primitive](*in_values)
        
        jax.util.safe_map(write, equation.invars, [out_values])
        
    return jax.util.safe_map(read, jaxpr.invars)

def inverse(functionPointer, inverse_registry):
    
    @jax.util.wraps(functionPointer)
    def wrapped_function(*args, **kwargs):
        
        function_jaxpr = jax.make_jaxpr(functionPointer)
        closed_jaxpr = function_jaxpr(*args, **kwargs)
        
        output = inverse_iterate_jaxpr(inverse_registry, closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        
        return output[0]
    
    return wrapped_function

def function(x):
    
    tan = jax.numpy.tanh(x)
    exp = jax.numpy.exp(tan)
    
    return exp
          
def test():
    
    function_jaxpr = jax.make_jaxpr(function)
    jaxpr = function_jaxpr(2.)
    
    print("jaxpr = ", jaxpr)
    print("---------------------------")
    
    inverse_registry = {jax.lax.exp_p: jax.numpy.log, jax.lax.tanh_p: jax.numpy.arctanh}
    
    function_inverse = inverse(function, inverse_registry)
    function_inverse_jaxpr = jax.make_jaxpr(function_inverse)
    
    result = function_inverse_jaxpr(function(2.))
    
    print("result = ", result)
    print("---------------------------")
    
def main():
    
    test()
    
if __name__ == "__main__":
    
    main()