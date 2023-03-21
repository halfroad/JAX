import jax

array = []

def log(x):
    
    # This statement breaches the rule of Pure Function, and it will not be executed
    array.append(x)
    
    ln_x = jax.numpy.log(x)
    ln_2 = jax.numpy.log(2.)
    
    return ln_x / ln_2

def test():
    
    log_jaxpr = jax.make_jaxpr(log)
    result = log_jaxpr(3.)
    
    print("result = ", result)
    print(array)
    
if __name__ == "__main__":
    
    test()