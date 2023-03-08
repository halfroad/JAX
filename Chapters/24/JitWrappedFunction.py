import jax
import time

def normalize(inputs):
    
    inputs = inputs.mean(0)
    
    return inputs / inputs.std(0)

def test():
    
    prng = jax.random.PRNGKey(15)
    inputs = jax.random.normal(prng, shape = (1024 ,1024))
    start = time.time()
    
    normalize(inputs)
    
    end = time.time()
    
    print(f"Time consumed: {end - start} seconds")
    
    normalize_jit = jax.jit(normalize)
    start = time.time()
    
    normalize_jit(inputs)
    
    end = time.time()
    
    print(f"Time consumed: {end - start} seconds")
    
if __name__ == "__main__":
    
    test()
    