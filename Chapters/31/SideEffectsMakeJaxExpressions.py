import jax

def print_log(x):
    
    print("print_test: ", x)
    
    x = jax.numpy.log(x)
    
    print("print_test: ", x)
    
    return x

def test():
    
    print_log_jaxpr = jax.make_jaxpr(print_log)
    result = print_log_jaxpr(3.)
    
    print("result = ", result)
    
if __name__ == "__main__":
    
    test()