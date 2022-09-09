def loss(parameters, x, y, apply_function):
    
    """

    Loss Function (Mean Squared Error, MSE):
    
    g(x) = [(f(x) - y)^2] / n
    
    """
    
    dotsProduct = apply_function(x, parameters)
    
    return jnp.mean(jnp.power(dotsProduct - y, 2.0))
    
    