import jax.numpy as jnp

def error_function(theta, x, y):
    
    """

    Ordinary Leat Squares
    
    """
    
    h_predicted = jnp.dot(x, theta)
    j_theta = (1. / (2 * m)) * jnp.dot(jnp.transpose(h_predicted), h_predicted)
    
    return j_theta