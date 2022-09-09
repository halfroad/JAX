import jax.numpy as jnp

from jax import random

def Dense(dense_shape = [2, 1]):
    
    def init(input_shape = dense_shape):
        
        prng = random.PRNGKey(17)
        
        w, b = random.normal(prng, shape = input_shape), random.normal(prng, shape = (input_shape[-1], ))
        
        return w, b
    
    def apply(inputs, parameters):
        
        weight, bias = parameters
        
        jnp.dot(inputs, weight) + bias
        
    return init, apply