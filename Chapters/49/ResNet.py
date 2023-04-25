import jax

def model():
    
    filter_number = 64
    filter_size = (3, 3)
    strides = (2, 2)
    
    jax.example_libraries.stax.Conv(filter_number, filter_size, strides)
    jax.example_libraries.stax.Conv(filter_number, filter_size, strides, padding = "SAME")
    
    window_shape = (3, 3)
    strides = (2, 2)
    
    jax.example_libraries.stax.AvgPool(window_shape = window_shape, strides = strides)
    
def train():
    
    model()
    
def main():
    
    train()