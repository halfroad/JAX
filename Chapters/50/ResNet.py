import jax.example_libraries.stax

def IdentityBlock(kernel_size, filters):
    
    kernel_size_ = kernel_size
    filters1, filters2 = filters
    
    # Generate a main path
    def make_main(inputs_shape):
        
        return jax.example_libraries.stax.serial(
            
            jax.example_libraries.stax.Conv(filters1, (1, 1), padding = "SAME"),
            jax.example_libraries.stax.BatchNorm(),
            jax.example_libraries.stax.Relu,
            
            jax.example_libraries.stax.Conv(filters2, (kernel_size_, kernel_size_), padding = "SAME"),
            jax.example_libraries.stax.BatchNorm(),
            jax.example_libraries.stax.Relu,
            
            # Adjust according to the inputs automatically
            jax.example_libraries.stax.Conv(inputs_shape[3], (1, 1), padding = "SAME"),
            jax.example_libraries.stax.BatchNorm()
        )
    
    Main = jax.example_libraries.stax.shape_dependent(make_layer = make_main)
    
    return jax.example_libraries.stax.serial(
        
        jax.example_libraries.stax.FanOut(2),
        jax.example_libraries.stax.parallel(Main,
                                            jax.example_libraries.stax.Identity),
                                            jax.example_libraries.stax.FanInSum,
                                            jax.example_libraries.stax.Relu
        )

def ConvolutionalBlock(kernel_size, filters, strides = (1, 1)):
    
    kernel_size_ = kernel_size
    filters1, filters2, filters3 = filters
    
    Main = jax.example_libraries.stax.serial(
        
        jax.example_libraries.stax.Conv(filters1, (1, 1), strides = strides, padding = "SAME"),
        jax.example_libraries.stax.BatchNorm(),
        jax.example_libraries.stax.Relu,
        
        jax.example_libraries.stax.Conv(filters2, (kernel_size_, kernel_size_), padding = "SAME"),
        jax.example_libraries.stax.BatchNorm(),
        jax.example_libraries.stax.Relu,
        
        jax.example_libraries.stax.Conv(filters3, (1, 1), strides = strides, padding = "SAME"),
        jax.example_libraries.stax.BatchNorm()
    )
    
    Shortcut = jax.example_libraries.stax.serial(
        jax.example_libraries.stax.Conv(filters3, (1, 1), strides, padding = "SAME")
    )
    
    return jax.example_libraries.stax.serial(
        
        jax.example_libraries.stax.FanOut(2),
        jax.example_libraries.stax.parallel(
            Main,
            Shortcut
        ),
        
        jax.example_libraries.stax.FanInSum,
        jax.example_libraries.stax.Relu)

def ResNet50(number_classes):
    
    return jax.example_libraries.stax.serial(
        
        jax.example_libraries.stax.Conv(64, (3, 3), padding = "SAME"),
        jax.example_libraries.stax.BatchNorm(),
        jax.example_libraries.stax.Relu,
        
        jax.example_libraries.stax.MaxPool((3, 3), strides = (2, 2)),
        
        ConvolutionalBlock(3, [64, 64, 256]),
        
        IdentityBlock(3, [64, 64]),
        IdentityBlock(3, [64, 64]),
        
        ConvolutionalBlock(3, [128, 128, 512]),
        
        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128,]),
        
        ConvolutionalBlock(3, [256, 256, 1024]),
        
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        
        ConvolutionalBlock(3, [512, 512, 2048]),
        
        IdentityBlock(3, [512, 512]),
        IdentityBlock(3, [512, 512]),
        
        jax.example_libraries.stax.AvgPool((7, 7)),
        
        jax.example_libraries.stax.Flatten,
        
        jax.example_libraries.stax.Dense(number_classes),
        
        jax.example_libraries.stax.LogSoftmax
    )
        
        
        