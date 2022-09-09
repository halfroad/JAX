import sys
import jax.numpy as jnp

from jax import random, grad

sys.path.append("../2.2.6.1/")
sys.path.append("../2.2.6.3/")

from IrisSpeciesRecognitionMultilayersPerceptronDatasetPreparation import prepare
from IrisSpeciesRecognitionMultilayersPerceptronModel import mlp, mlp_loss

def train(data, targets):
    
    prng = random.PRNGKey(17)
    
    a0 = random.normal(prng, shape = [4, 5])
    b0 = random.normal(prng, shape = (5,))
    
    a1 = random.normal(prng, shape = [5, 10])
    b1 = random.normal(prng, shape = (10,))
    
    parameters = [a0, b0, a1, b1]
    learning_rate = 2.17e-4
    
    for i in range(20000):
        
        loss = mlp_loss(parameters, data, targets)
        
        if i % 100 == 0:
            
            predictions = mlp(data, parameters)
            classification = jnp.argmax(predictions, axis = 1)
            
            target = jnp.argmax(targets, axis = 1)
            accuracy = jnp.sum(classification == target) / len(target)
            
            print("i:", i, "loss:", loss, "accuracy:", accuracy)
            
        gradient_parameters = grad(mlp_loss)(parameters, data, targets)
        
        parameters = [
            
            (p - g * learning_rate) for p, g in zip(parameters, gradient_parameters)
            ]
        
    predictions = mlp(data, parameters)
    classification = jnp.argmax(predictions, parameters)
    
    target = jnp.argmax(targets, axis = 1)
    accuracy = jnp.sum(classification == target) / len(target)
    
    print("Accuracy:", accuracy)
    
def main():
    
    (data, targets) = prepare()
    
    train(data, targets)
    
if __name__ == "__main__":
    
    main()
