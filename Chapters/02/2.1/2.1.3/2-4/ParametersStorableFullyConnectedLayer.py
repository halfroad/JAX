import jax.numpy as jnp

from jax import random

def Dense17(dense_shape = [2, 1]):
    
    def init(input_shape = dense_shape):
        
        prng = random.PRNGKey(17)
        
        weight, bias = random.normal(prng, shape = dense_shape), random.normal(prng, shape = (dense_shape[-1], ))
        
        return (weight, bias)
    
    def apply(inputs, parameters):
        
        weight, bias = parameters
        
        return jnp.dot(inputs, weight) + bias

    return init, apply

def Dense18(dense_shape = [2, 1]):
    
    def init(input_shape = dense_shape):
        
        prng = random.PRNGKey(18)
        
        weight, bias = random.normal(prng, shape = input_shape), random.normal(prng, shape = (input_shape[-1], ))
        
        return (weight, bias)
    
    def apply(inputs, parameters):
        
        weight, bias = parameters
        
        return jnp.dot(inputs, weight) + bias
    
    return init, apply

def main():
    
    inputs = jnp.array([[1.7, 1.7],
                        [2.14, 2.14]])
    init, apply = Dense17()
    parameters = init()
    
    dotted = apply(inputs, parameters)
    
    print("dotted = {}".format(dotted))
    
    init, apply = Dense18()
    parameters = init()
    
    dotted = apply(inputs, parameters)
    
    print("dotted = {}".format(dotted))
    
if __name__ == "__main__":
    
    main()
    
        