import jax

def pool(features, size = 2, stride = 2):
        
    features_shape = features.shape
    
    print("features_shape = ", features_shape)
    
    height = features_shape[0]
    width = features_shape[1]
    
    print(f"height = {height}, width = {width}")
    
    padding_height = round((height - size + 1) / stride)
    padding_width = round((width - size + 1) / stride)
    
    print(f"padding_height = {padding_height}, padding_width = {padding_width}")
    
    outputs = jax.numpy.zeros(shape = (padding_height, padding_width))
    
    print("outputs.shape = ", outputs.shape)
    
    output_height = 0
    
    for i in jax.numpy.arange(0, height, stride):
        
        output_width = 0
        
        for j in jax.numpy.arange(0, width, stride):
            
            outputs = outputs.at[output_height, output_width].set(jax.numpy.max(features[i: i + size, j: j + size]))
            
            output_height = output_height + 1
            output_width = output_width + 1
            
    return outputs

def test():
    
    prng = jax.random.PRNGKey(15)
    
    image = jax.random.normal(key = prng, shape = (10, 10))
    pooled = pool(image)
    
    print("pooled.shape = ", pooled.shape)
    
if __name__ == "__main__":
    
    test()
    