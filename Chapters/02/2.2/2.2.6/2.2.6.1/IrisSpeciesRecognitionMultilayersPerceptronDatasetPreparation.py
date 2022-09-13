from sklearn.datasets import load_iris
from jax import random

import jax.numpy as jnp

def load():
    
    iris = load_iris()
    
    data = jnp.float32(iris.data)
    targets = jnp.float32(iris.target)
    
    prng = random.PRNGKey(17)
    
    data = random.permutation(random.PRNGKey(17), data, independent = True)
    targets = random.permutation(random.PRNGKey(17), targets, independent = True)
    
    return (data, targets)
    
def one_hot_nojit(x, k = 10, dtype = jnp.float32):
    
    """

    Create an one-hot encoding of x of size k.
    
    """
    
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def prepare():
    
    (data, targets) = load()
    
    targets = one_hot_nojit(targets)
    
    return (data, targets)

array = jnp.array([0, 1, 2, 5, 8, 1, 9, 2])

print(array.shape)

array = one_hot_nojit(array)

print(array.shape)
print(array)