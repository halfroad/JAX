import jax

def batch_normalize(inputs, beta = 0.25, epsilon = 1e-9, gamma = 0.9):
    
    u = jax.numpy.mean(inputs, axis = 0)
    std = jax.numpy.sqrt(inputs.var(axis = 0) + epsilon)
    y = (inputs - u) / std
    
    y_hat = gamma * y + beta
    
    return y_hat

def test():
    
    prng = jax.random.PRNGKey(15)
    inputs = jax.random.normal(key = prng, shape = (10,))
    
    print("inputs = ", inputs)
    
    y_hat = batch_normalize(inputs);
    
    print("y_hat = ", y_hat)
    
if __name__ == "__main__":
    
    test();