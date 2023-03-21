import jax

@jax.jit
def loop(i):
    
    return i + 1

#@jax.jit
def isolate_for_jit(x, n):
    
    i = 0
    
    while i < n:
        
        i = loop(i)
        
    return x + i

def test():
    
    result = isolate_for_jit(10, 20)
    
    print("result = ", result)
    
if __name__ == "__main__":
    
    test()

