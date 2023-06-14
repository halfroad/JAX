import jax

'''

f(x) = α x (eˣ - 1) x θ

'''
def selu(x, alpha = 1.67, theta = 1.05):

    return alpha * jax.numpy.where(x > 0, x, jax.numpy.exp(x) - 1) * theta
    
def main():

    prng = jax.random.PRNGKey(10);
    inputs = jax.random.normal(key = prng, shape = (10,))
    
    print(f"inputs = {inputs}")
    
    outputs = selu(inputs)
    
    print(f"outputs = {outputs}")
    
if __name__ == "__main__":

    main()
