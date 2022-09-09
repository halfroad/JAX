import jax.numpy as jnp

from jax import random

def Dense(dense_shape = [2, 1]):
    
    def init(input_shape = dense_shape):
        
        prng = random.PRNGKey(17)
        
        w, b = random.normal(prng, shape = input_shape), random.normal(prng, shape = (input_shape[-1], ))
        
        return (w, b)
    
    def apply(inputs, parameters):
        
        weight, bias = parameters
        
        return jnp.dot(inputs, weight) + bias
        
    return init, apply

def main():
    
    init, apply = Dense()
    
    inputs = jnp.array([[1.7, 1.7],
                        [2.14, 2.14]
                        ])
    parameters = init()
    
    dotted = apply(inputs, parameters)
    
    print("dotted = {}".format(dotted))
    
if __name__ == "__main__":
    
    main()