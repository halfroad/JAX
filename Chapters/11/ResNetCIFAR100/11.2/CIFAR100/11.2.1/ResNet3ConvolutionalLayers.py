from jax.example_libraries import stax


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


