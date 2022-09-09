from sklearn.datasets import load_iris
# grad: Gradient
from jax import random, grad
from collections import namedtuple

import jax.numpy as jnp

def load():
    
    iris = load_iris()
    
    data = jnp.float32(iris.data)
    targets = jnp.float32(iris.target)
    
    return (data, targets)

def Dense(dense_shape = [4, 1]):
    
    def init(input_shape = dense_shape):
        
        prng = random.PRNGKey(17)
        weight, bias = random.normal(prng, shape = input_shape), random.normal(prng, shape = (input_shape[-1], ))
        
        return (weight, bias)
    
    def apply(inputs, parameters):
        
        weight, bias = parameters
        
        return jnp.dot(inputs, weight) + bias
    
    return init, apply

def loss_function(parameters, x, y, apply_function):
    
    """

    Loss Function (Mean Squared Error, MSE)
    
    g(x) = (f(x) - y) ** 2
    g(x) = [(f(x) - y)^2] / n
    
    """
    
    dotsProduct = apply_function(x, parameters)
    powers = jnp.power(dotsProduct - y, 2.0)
    
    return jnp.mean(powers)

def train(configuration, data, targets):
    
    init, apply = Dense()
    
    parameters = init()
    
    for i in range(configuration.epochs):
        
        # Compute the loss
        loss = loss_function(parameters, data, targets, apply)
        
        if i % 100 == 0:
            
            print("Loss at {} is {}.".format(i, loss))
            
        # Compute and update the gradient algorithm
        gradient_parameters = grad(loss_function)(parameters, data, targets, apply)
        
        parameters = [
            
            # For each parameter, plus the value of learning_rate * [negative]derivative
            (p - g * configuration.learning_rate) for p, g in zip(parameters, gradient_parameters)
            ]
        
    print(f"{i}: {configuration.epochs}, loss: {loss}")
    

def acquire_configurations():
    
    Confguration = namedtuple("Confguration", ["learning_rate", "epochs"])
    
    configuration = Confguration(learning_rate = 0.005, epochs = 1000)
    
    return configuration

def main():
    
    (data, targets) = load()
    configuration = acquire_configurations()
    
    train(configuration, data, targets)
    
if __name__ == "__main__":
    
    main()
        
    