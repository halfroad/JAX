import jax.numpy as jnp

from jax import random

array1 = jnp.arange(10)

print(array1)

prng = random.PRNGKey(17)
array1 = random.permutation(prng, array1, independent = True)

print(array1)

array2 = jnp.arange(12).reshape(3, 4)

print(array2)

array2 = random.permutation(prng, array2, independent = True)

print(array2)