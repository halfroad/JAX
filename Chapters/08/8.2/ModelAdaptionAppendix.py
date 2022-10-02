import random

import jax


def predict(parameters, inputs):

    outputs = jax.numpy.dot(parameters, inputs)

    return outputs

prng = jax.random.PRNGKey(17)
signle_random_flattened_images1 = jax.random.normal(prng, (28 * 28,))
random_flattened_images2 = jax.random.normal(prng, (10, 28 * 28,))
w = jax.random.normal(prng, (256, 784))

# predictions = predict(w, inputs = signle_random_flattened_images1)

# Error
# predictions = predict(w, inputs = random_flattened_images2)

vmap_predict = jax.vmap(predict, [None, 0])
predictions = vmap_predict(w, random_flattened_images2)

print(predictions)
