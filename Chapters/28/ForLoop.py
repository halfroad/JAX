import jax

def body_func(i, x):
    
    return i + x + 2

def test():
    
    init_val = 0
    lower = 0
    upper = 100
    
    body_function = lambda i, x: x + i
    
    result = jax.lax.fori_loop(lower, upper, body_function, init_val)
    print("Result = ", result)
    
    result = jax.lax.fori_loop(lower, upper, body_func, init_val)
    print("Result = ", result)
    
if __name__ == "__main__":
    
    test()