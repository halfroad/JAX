import jax

def condition_function(x):
    
    return x < 15

def body_function(x):
    
    return x + 1

def test():
    
    init_value = 5
    
    result = jax.lax.while_loop(condition_function, body_function, init_value)
    
    print("result = ", result)
    
if __name__ == "__main__":
    
    test()