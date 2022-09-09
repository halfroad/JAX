import jax.numpy as jnp

from jax import random

def Dense(dense_shape = [2, 1]):
    
    # 17 is the seed of random, can be any
    prng = random.PRNGKey(17)
    weight = random.normal(prng, shape = dense_shape)
    
    print("dense_shape = {}".format(dense_shape))
    print("dense_shape[-1] = {}".format(dense_shape[-1]))
    
    # random.normal: Normal Distribution
    # random.uniform: Genuine/Real numbers
    # random.choice: select from a list randomly
    bias = random.normal(prng, shape = (dense_shape[-1],))
    
    print("weight = {}, bias = {}".format(weight, bias))
    
    parameters = [weight, bias]
    
    print("parameters = {}".format(parameters))
    
    # apply() is a characteristic feature of python, so-called intrinsic function
    def apply(inputs, parameters = parameters):
        
        w, b = parameters
        
        return jnp.dot(inputs, w) + b
    
    return apply

def start():
    
    prng = random.PRNGKey(18)
    dense_shape = [2, 1]
    
    weight = random.normal(prng, shape = dense_shape)
    bias = random.normal(prng, shape = (dense_shape[-1], ))
    
    print("weight = {}, bias = {}".format(weight, bias))
    
    array = jnp.array([[1.7, 1.7],
                       [2.14, 2.14]])
    parameters = [weight, bias]
    
    print("parameters = {}".format(parameters))
    
    result = Dense()(array, parameters)
    
    print("Result = {}".format(result))
    
start()