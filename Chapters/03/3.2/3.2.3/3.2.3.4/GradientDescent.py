import jax.numpy as jnp

from GradientFunction import gradient_function

def gradient_descent(x, y, alpha):
    
    """

    Gradient Descent
    
    """
    
    # [2, 1] is the shape of theta parameter
    theta = jnp.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, x, y)
    
    while not jnp.all(jnp.absolute(gradient) <= 1e-5):
        
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, x, y)
        
    return theta