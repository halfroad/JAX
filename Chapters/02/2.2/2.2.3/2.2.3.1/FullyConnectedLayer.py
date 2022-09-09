import jax.numpy as jnp

from jax import random

def Dense(dense_shape = [4, 1]):
    
    """

    1. 1st step: Use of Fully Connected Layer

    """
    
    def init(input_shape = dense_shape):
        
        prng = random.PRNGKey(17)
        weight, bias = random.normal(prng, shape = input_shape), random.normal(prng, shape = (input_shape[-], ))
        
        return (weight, bias)
    
    def apply(inputs, parameters):
        
        weight, bias = parameter
        
        return jnp.dot(inputs, weight) + bias
    
    return init, apply