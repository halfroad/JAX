import jax
import functools

@jax.jit
def function(x, negative):
    
    return -x if negative else x

@functools.partial(jax.jit, static_argnums = (1,))
def function1(x, negative):
    
    return -x if negative else x

def test():
    
   # print(function(1, True))
    print(function1(1, True))
    
if __name__ == "__main__":
    
    test()
