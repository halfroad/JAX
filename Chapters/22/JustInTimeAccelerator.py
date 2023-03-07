import time
import jax

def selu(inputs, alpha = 1.67, theta = 1.05):
    
    return jax.numpy.where(inputs > 0, inputs, alpha * jax.numpy.exp(inputs) - alpha) * theta

@jax.jit
def selu1(inputs, alpha = 1.67, theta = 1.05):
    
    return jax.numpy.where(inputs > 0, inputs, alpha * jax.numpy.exp(inputs) - alpha) * theta

def test():
    
    key = jax.random.PRNGKey(15)
    inputs = jax.random.normal(key = key, shape = (1000000,))
    
    start = time.time()
    
    selu(inputs = inputs)
    
    end = time.time()
    
    print(f"Time consumed by selu: %.2f seconds" % (end - start))
    
    jit_selu = jax.jit(selu)
    
    start = time.time()
    
    jit_selu(inputs)
    
    end = time.time()
    
    print(f"Time consumed by jit_selu: %.2f seconds" % (end - start))
    
    start = time.time()
    
    selu1(inputs)
    
    end = time.time()
    
    print("Time consumed by selu1: %.2f seconds" % (end - start))
    
if __name__ == "__main__":
    
    test()
    
    
    

    
    