import jax

def convolve(inputs, weights, stride_size = 2):
    
    outputs = []
    
    for i in range(inputs.shape[-1] - stride_size):
        
        slices = inputs[i: i + stride_size + 1]
        dotted = jax.numpy.dot(slices, weights)
        
        outputs.append(dotted)
        
    outputs = jax.numpy.array(outputs)
    
    return outputs

def test():
    
    inputs = jax.numpy.arange(5)
    # inputs = jax.numpy.stack([inputs, inputs])
    
    weights = jax.numpy.array([2., 3., 4.])
    # weights = jax.numpy.stack([weights, weights])
    
    outputs = convolve(inputs, weights)
    
    print("outputs = ", outputs)
    
    print("------------------------------------------")
    
    auto_batch_convolve = jax.vmap(convolve)
    
    inputs = jax.numpy.stack([inputs, inputs])
    weights1 = jax.numpy.stack([weights, weights])
    
    outputs = auto_batch_convolve(inputs, weights1)
    
    print("outputs = ", outputs)
    
    print("------------------------------------------")
    
    auto_batch_convolve_v2 = jax.vmap(convolve, in_axes = 1, out_axes = 1)
    
    inputs = jax.numpy.transpose(inputs)
    weights1 = jax.numpy.transpose(weights1)
    
    outputs = auto_batch_convolve_v2(inputs, weights1)
    
    print("outputs = ", outputs)
    
    print("------------------------------------------")
    
    auto_batch_convolve_v3 = jax.vmap(convolve, in_axes = [0, None])
    weights2 = jax.numpy.stack([weights, weights], axis = 0)
    
    outputs = auto_batch_convolve_v3(inputs, weights2)
    
    print("outputs = ", outputs)
    
if __name__ == "__main__":
    
    test()
    
                               