import jax

def Dense(input_shape = (2, 1)):

    key = jax.random.PRNGKey(10)
    
    weights = jax.random.normal(key = key, shape = input_shape)
    biases = jax.random.normal(key = key, shape = (input_shape[-1],))
    
    params = [weights, biases]
    
    def apply_function(inputs):
    
        weights, biases = params
        dotted = jax.numpy.dot(inputs, weights) + biases
        
        return dotted
        
    return apply_function

def test():

    array = [[1.1, 1.8], [1.2, 1.7]]
    inputs = jax.numpy.array(array)
    
    dense = Dense()(inputs)
    
    print(dense)
    
if __name__ == "__main__":

    test()
