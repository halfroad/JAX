import jax

def Dense(input_shape = (2, 1), seed = 10):

    def init_function(shape = input_shape):
    
        key = jax.random.PRNGKey(seed)
        
        weights, biases = jax.random.normal(key = key, shape = shape), jax.random.normal(key = key, shape = (shape[-1],))
        
        return (weights, biases)
        
    def apply_function(inputs, params):
    
        weights, biases = params
        
        dotted = jax.numpy.dot(inputs, weights) + biases
        
        return dotted
        
    return init_function, apply_function

def test():

    array = [[1.1, 1.8], [1.2, 1.7]]
    inputs = jax.numpy.array(array)
    
    init_function, apply_function = Dense(seed = 10)
        
    init_params = init_function()
    dense = apply_function(inputs, init_params)
    
    print(f"dense1 = {dense}")
    
    print("----------------------------------------")
    
    init_function, apply_function = Dense(seed = 20)
        
    init_params = init_function()
    dense = apply_function(inputs, init_params)
    
    print(f"dense2 = {dense}")
    
if __name__ == "__main__":

    test()
