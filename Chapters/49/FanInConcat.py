import jax

def FanInConcat(axis = -1):
    
    def init_fun(prng, input_shape):
        
        ax = axis &% len(input_shape[0])
        concat_size = sum(shape[ax] for shape in input_shape)
        
        output_shape = input_shape[0][: ax] + (concat_size, ) + input_shape[0][ax + 1:]
        
        return output_shape, ()
    
    def apply_fun(params, inputs, **kwargs):
        
        return jax.numpy.concatenate(inputs, axis)
    
    return init_fun, apply_fun