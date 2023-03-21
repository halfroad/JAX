import jax

def convolve(inputs, weights, stride_size = 2):
    
    outputs = []
    
    # i in [0, 1, 2]
    # len(inputs) = 5
    for i in range(len(inputs) - stride_size):
        
        # inputs[0: 3], inputs[1: 4], inputs[2, 5]
        slices = inputs[i: i + stride_size + 1]
        dotted = jax.numpy.dot(slices, weights)
        outputs.append(dotted)
        
    return jax.numpy.array(outputs)

def batched_convolve(inputs, weights):
    
    outputs = []
    
    for i in range(inputs.shape[0]):
        
        outputs.append(convolve(inputs[i], weights[i]))
    
    outputs = jax.numpy.stack(outputs)
    
    return outputs

def vectorized_convolve(inputs, weights, stride_size = 2):
    
    outputs = []
    
    # inputs.shape[-1] = 5
    for i in range(inputs.shape[-1] - stride_size):
        
        slices = inputs[:, i: i + stride_size + 1]
        dotted = slices * weights
        accumulation = jax.numpy.sum(dotted, axis = 1)
        
        outputs.append(accumulation)
    
    outputs = jax.numpy.stack(outputs, axis = 1) 
        
    return outputs

def vectorized_convolve_v1(inputs, weights, stride_size = 2):
        
    outputs = []
        
    for i in range(0, inputs.shape[-1] - stride_size):
            
        slices = inputs[:, i: i + stride_size + 1]
        dotted = slices @ weights.T
        outputs.append(dotted)
            
    jax.numpy.stack(outputs, axis = 1)
        
    return outputs

def test():
    
    # [0, 1, 2, 3, 4]
    inputs = jax.numpy.arange(5)
    weights = jax.numpy.array([2., 3., 4.])
    
    # outputs = convolve(inputs, weights)
    
    # print(f"inputs = {inputs}, weights = {weights}, outputs = ", outputs)
    
    inputs = jax.numpy.stack([inputs, inputs])
    weights = jax.numpy.stack([weights, weights])
    
    '''
    
    outputs = batched_convolve(inputs, weights)
        
    print(f"inputs = {inputs}\n, weights = {weights},\n outputs = ", outputs)
    
    print("------------------------------------------")
    
    outputs = vectorized_convolve(inputs, weights)
        
    print(f"inputs = {inputs}, \nweights = {weights},\n outputs = ", outputs)
    
    print("------------------------------------------")
    
    '''
    
    outputs = vectorized_convolve_v1(inputs, weights)
        
    print(f"inputs = {inputs}, \nweights = {weights},\n outputs = ", outputs)
    
    
    
if __name__ == "__main__":
    
    test()