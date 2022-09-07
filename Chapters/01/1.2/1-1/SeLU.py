import jax.numpy as jnp		# Import the computing package of NumPy
from jax import random		# Import the random package

# Implementation of SeLU function
def selu(x, alpha = 1.67, lmbda = 1.05):

	return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

key = random.PRNGKey(17)		# Generate a fixed number 17 as a key
x = random.normal(key, (5,))

print(selu(x))
