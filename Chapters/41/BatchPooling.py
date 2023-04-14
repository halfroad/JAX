import jax

def batch_pool(features, size = 2, stride = 2):
    
    assert features.ndim == 4, print("The inputs dimension should be 4.")
    
    # Dimension transpose
    features = jax.numpy.einsum("bhwc -> bchw", features)
    
    # Pool the singular input.
    def pool(features, size = size, stride = stride):
        
        channels = features.shape[0]
        
        height = features.shape[1]
        width = features.shape[2]
        
        padding_height = round((height - size + 1) / stride)
        padding_width = round((width - size + 1) / stride)
        
        outputs = jax.numpy.zeros(shape = (channels, padding_height, padding_width))
        
        print("outputs.shape = ", outputs.shape)
        
        for channel in range(channels):
            
            output_height = 0
            
            for i in jax.numpy.arange(0, height, stride):
                
                output_width = 0
                
                for j in jax.numpy.arange(0, width, stride):
                    
                    outputs = outputs.at[channel, output_height, output_width].set(jax.numpy.max(features[channel, i: i + size, j: j + size]))
                    output_width = output_width + 1
                    
                output_height = output_height + 1
                
        return outputs
    
    batch_pool_vmap = jax.vmap(pool)
    
    batch_pool_outputs = batch_pool_vmap(features)
    
    batch_pool_outputs = jax.numpy.einsum("bchw -> bhwc", batch_pool_outputs)
    
    return batch_pool_outputs

def test():
    
    prng = jax.random.PRNGKey(15)
    
    image = jax.random.normal(key = prng, shape = (2, 10, 10, 3))
    batch_pool_outputs = batch_pool(image)
    
    print("batch_pool_outputs.shape = ", batch_pool_outputs.shape)
    
if __name__ == "__main__":
    
    test()
    