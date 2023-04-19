import jax

def multiply_add(a, b, c):
    
    d = jax.lax.mul(a, b)
    e = jax.lax.add(d, c)
    
    return e

def main():
    
    a = 2
    b = 4
    c = 6
    
    e = multiply_add(a, b, c)
    
    print(f"e = {e}")
    
if __name__ == "__main__":
    
    main()