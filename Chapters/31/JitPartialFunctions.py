import jax
import time

#@jax.jit
def loop(i):
    
    return i + 1

def isolate_for_jit(x, n):
    
    i = 0
    
    while i < n:
        
        i = loop(i)
        # don't jit the simle method
        # loop_jit = jax.jit(loop)
        # i = loop_jit(i)
        
    return x + i

def test():
    
    result = isolate_for_jit(10, 2000)
    print("result = ", result)
    
if __name__ == "__main__":
    
    begin = time.time()
    test()
    end = time.time()
    
    print(f"Time consumed: {end - begin} seconds")
        