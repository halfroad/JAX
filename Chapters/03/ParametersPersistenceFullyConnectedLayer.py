import jax

def Dense(inputs_shape = (2, 1)):

    def init_function(shape = inputs_shape):
    
        key = jax.random.PRNGKey(10)
        
        weights, biases = jax.random.normal(key = key, shape = shape), jax.random.normal(key = key, shape = (shape[-1],))
        
        return (weights, biases)
        
    def apply_function(inputs, params):
    
        weights, biases = params
        
        dotted = jax.numpy.dot(inputs, weights) + biases
        
        return dotted
        
    return init_function, apply_function
    
def test():

    init_function, apply_function = Dense()
    
    init_params = init_function()
    
    array = [[1.1, 1.8], [1.2, 1.7]]
    inputs = jax.numpy.array(array)
    
    result = apply_function(inputs, init_params)
    
    print(f"init_params = {init_params}, result = {result}")
    
if __name__ == "__main__":

    test()
