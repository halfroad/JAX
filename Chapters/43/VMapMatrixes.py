import jax

@jax.jit
def predict(weights, inputs):
    
    outputs = jax.numpy.dot(weights, inputs)
    
    return outputs

def multiplify(key):
    
    single_random_flattened_images = jax.random.normal(key = key, shape = (28 * 28,))
    random_flattened_images = jax.random.normal(key = key, shape = (10, 28 * 28))
    
    weights = jax.random.normal(key = key, shape = (256, 784))
    
    # return predict(weights, single_random_flattened_images)
    # return predict(weights, random_flattened_images)
    vmap_predict = jax.vmap(predict, [None, 0])
    
    return vmap_predict(weights, random_flattened_images)
    
def test():
    
    key = jax.random.PRNGKey(15)
    
    print(multiplify(key))
    
if __name__ == "__main__":
    
    test();