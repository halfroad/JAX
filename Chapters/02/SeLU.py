import jax

def selu(x, alpha = 1.67, lambda_ = 1.05):
    
    return jax.numpy.where(x > 0, x, alpha * (jax.numpy.exp(x) - 1) * lambda_)

def main():
    
    key = jax.random.PRNGKey(15)
    
    inputs = jax.radmon.normal(key, (10,))
    
    print(f"inputs = {inputs}")
    
    outputs = selu(inputs)
    
    print(f"outputs = {outputs}")
    
if __name__ == "__main__":
    
    main()