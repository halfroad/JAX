import jax
import jax.numpy as jnp
import timeit

def f(x):
    
    return x * x * x

def main():
    
    start = timeit.default_timer()
    
    x = 1.0
    
    r = f(x)
    
    print("r = {}".format(r))
    
    d_f = jax.grad(f)
    d_d_f = jax.grad(d_f)
    d_d_d_f = jax.grad(d_d_f)
    
    print(d_f(x))
    print(d_d_f(x))
    print(d_d_d_f(x))
    
    # Sum firstly
    d_f = jax.grad(lambda x: sum1(x))
    
    x = jnp.linspace(1, 5, 5)
    
    print(x)
    
    r = d_f(x)
    
    print(r)
    
    import_module = "import random"
    print(timeit.default_timer() - start)
    
def sum1(x):
    
    y = f(x)
    r = jnp.sum(y)
    
    return r
    
if __name__ == "__main__":
    
    main()