import jax
import functools

'''
def GeneralConvolution(dimension_numbers, out_channel, filter_shape, strides = None, padding = "VALID", weights_init = None, biases_init = normal(1e-6)):
    
    # ...
    return init_fun, apply_fun

Conv = functools.partial(GeneralConvolution, ("NHWC", "HWIO", "NHWC"))
'''



def model():
    
    jax.example_libraries.stax.Conv();
    
def train():
    
    model()
    
def main():
    
    train()