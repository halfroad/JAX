import jax
import jax.numpy as jnp

def softmax(x, axis = -1):
    
    unnormalized = jnp.exp(x)
    
    return unnormalized / unnormalized.sum(axis, keepdims = True)

array = jnp.array([[3, 1, -3]])

probabilities = softmax(array)

print("probabilities = {}".format(probabilities))

probabilities = jax.nn.softmax(array)

print("probabilities = {}".format(probabilities))
