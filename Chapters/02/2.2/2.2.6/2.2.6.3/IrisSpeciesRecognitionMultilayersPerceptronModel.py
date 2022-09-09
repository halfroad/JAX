import jax
import jax.numpy as jnp
import sys

sys.path.append("../2.2.6.2/")

from IrisSpeciesRecognitionMultilayersPerceptronComponents import Dense, selu, softmax, cross_entropy

def mlp(x, parameters):
    
    """

    Multilayers Perceptron
    
    """
    
    # Import parameters
    a0, b0, a1, b1 = parameters
    
    # Hidden Layer
    x = Dense()(x, [a0, b0])
    
    # Activation Function
    x = jax.nn.tanh(x)
    
    # Output Layer
    x = Dense()(x, [a1, b1])
    
    # Softmax Layer
    x = softmax(x, axis = -1)
    
    return x

def mlp_loss(parameters, x, genuines):
    
    predicted = mlp(x, parameters)
    loss = cross_entropy(genuines, predicted)
    
    return jnp.mean(loss)
    
    
    