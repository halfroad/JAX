from jax.example_libraries import stax

def ConvBlock(kernel_size, filters, strides = (1, 1)):

    kernel_size_ = kernel_size

    # Set the numbers of convolutional kernel.
    filters1, filters2, filters3 = filters

    # Generate the main path
    Main = stax.serial(

        stax.Conv(filters1, (1, 1), strides, padding = "SAME"),
        stax.BatchNorm(),
        stax.Relu,

        stax.Conv(filters2, (kernel_size_, kernel_size_), padding = "SAME"),
        stax.BatchNorm(),
        stax.Relu,

        stax.Conv(filters3, (1, 1), padding = "SAME"),
        stax.BatchNorm()
    )

    Shortcut = stax.serial(

        stax.Conv(filters3, (1, 1), strides, padding = "SAME"),
        stax.BatchNorm()
    )

    return stax.serial(

        stax.FanOut(2),
        stax.parallel(Main, Shortcut),
        stax.FanInSum,
        stax.Relu
    )

def IdentityBlock(kernel_size, filters):

    kernel_size_ = kernel_size
    filters1, filters2 = filters

    # Generate the main path at first, here the dynamic self-assigned dimensions is used to adjust the dimensions
    def make_main(input_shape):

        return stax.serial(

            stax.Conv(filters1, (1, 1), padding = "SAME"),
            stax.BatchNorm(),
            stax.Relu,

            stax.Conv(filters2, (kernel_size_, kernel_size_), padding = "SAME"),
            stax.BatchNorm(),
            stax.Relu,

            # Adjust the dimensions dynamically relies on the input shape
            stax.Conv(input_shape[3], (1, 1), padding = "SAME"),
            stax.BatchNorm()
        )

    # Explicitly pass the size of dynamic input dimensions required by the model
    Main = stax.shape_dependent(make_main)

    # Combine the different computation channels
    return stax.serial(stax.FanOut(2),
                       stax.parallel(Main, stax.Identity),
                       stax.FanInSum,
                       stax.Relu)


def ResNet50(number_classes: int):

    return stax.serial(

        stax.Conv(64, (3, 3), padding = "SAME"),
        stax.BatchNorm(),
        stax.Relu,

        stax.MaxPool((3, 3), strides = (2, 2)),

        ConvBlock(3, [64, 64, 256]),

        IdentityBlock(3, [64, 64]),
        IdentityBlock(3, [64, 64]),

        ConvBlock(3, [128, 128, 512]),

        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),

        ConvBlock(3, [256, 256, 1024]),

        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),

        ConvBlock(3, [512, 512, 2048]),

        IdentityBlock(3, [512, 512]),
        IdentityBlock(3, [512, 512]),

        stax.AvgPool((7, 7)),

        stax.Flatten,
        stax.Dense(number_classes),
        stax.LogSoftmax
    )
