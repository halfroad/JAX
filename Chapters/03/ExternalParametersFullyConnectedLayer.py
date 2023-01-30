import jax

def Dense(inputs_shape = (2, 1)):

    key = jax.random.PRNGKey(10)
    
    weights = jax.random.normal(key = key, shape = inputs_shape)
    biases = jax.random.normal(key = key, shape = (inputs_shape[-1],))
    
    params = [weights, biases]
    
    def init_params_function():
    
        return params
        
    def apply_function(inputs, params = params):
    
        weights, biases = params
        dotted = jax.numpy.dot(inputs, weights) + biases
        
        return dotted
        
    return init_params_function, apply_function
    
def test():

    key = jax.random.PRNGKey(15)
    inputs_shape = (2, 1)
    
    weights = jax.random.normal(key = key, shape = inputs_shape)
    biases = jax.random.normal(key = key, shape = inputs_shape)
    
    params = [weights, biases]
    
    array = [[1.1, 1.8], [1.2, 1.7]]
    inputs = jax.numpy.array(array)
    
    init_params_function, apply_function = Dense()
    dense = apply_function(inputs, params)
    
    print(dense)
    
if __name__ == "__main__":

    test()
