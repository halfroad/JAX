import jax.numpy as jnp

def gradient_function(theta, x, y):
    
    """

    Gradient Function
    
    """
    
    h_predicted = jnp.dot(x, theta) - y
    
    return (1. / m) * jnp.dot(jnp.transpose(x), h_predicted)