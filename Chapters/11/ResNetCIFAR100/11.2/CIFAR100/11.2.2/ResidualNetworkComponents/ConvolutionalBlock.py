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
