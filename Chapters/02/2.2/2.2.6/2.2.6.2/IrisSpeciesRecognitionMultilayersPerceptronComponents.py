from jax import random

import jax.numpy as jnp

# Fully Connected Layer
def Dense(dense_shape = [1, 1]):
    
    prng = random.PRNGKey(17)
    
    weight = random.normal(prng, shape = dense_shape)
    bias = random.normal(prng, shape = (dense_shape[-1],))
    
    parameters = [weight, bias]
    
    def apply(inputs, parameters = parameters):
        
        weight, bias = parameters
        
        return jnp.dot(inputs, weight) + bias
    
    return apply

def selu(x, alpha = 1.67, lmbda = 1.05):
    
    """

    Activation Function: SeLU
    
    f(x) = alpha * (e^x - 1) * lmbda
    
    """
    
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


def softmax(x, axis = -1):
    
    """
    
    Softmax Function
    
    S(i) = jnp.exp(Vi) / jnp.sum(jnp.exp(Vi))
    
    """
    
    unnormalized = jnp.exp(x)
    
    return unnormalized / unnormalized.sum(axis, keepdims = True)

def cross_entropy(genuines, predicted):
    
    """

    Loss: Cross-Entropy Function
    
    H(p, q) = - jnp.sum[i = 1, i = n](p(Xi)) * log(q(Xi))
    
    """
    
    genuines = jnp.array(genuines)
    predicted = jnp.array(predicted)
    
    difference = - jnp.sum(genuines * jnp.log(predicted + 1e-7), axis = -1)
    
    return difference
    
    