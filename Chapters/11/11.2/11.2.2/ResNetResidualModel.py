import sys
import jax

from jax.example_libraries import stax
from jax.example_libraries.stax import Conv, BatchNorm, Relu, FanOut, FanInSum, Identity, MaxPool, AvgPool, Flatten, Dense, LogSoftmax

def IdentityBlock(kernel_size, filters):

    kernel_size_ = kernel_size
    filters1, filters2 = filters

    # Generate a main shortcut
    def make_main(input_shape):

        return stax.serial(

            Conv(filters1, (1, 1), padding = "SAME"),
            BatchNorm(),
            Relu,

            Conv(filters2, (kernel_size_, kernel_size_), padding = "SAME"),
            BatchNorm(),
            Relu,

            # Reshape the dimensions dynamically
            Conv(input_shape[3], (1, 1), padding = "SAME"),
            BatchNorm()
        )

    # Explicit to pass the dynamic configuration of input dimensions
    Main = stax.shape_dependent(make_main)

    # Combine the different computing channels
    return stax.serial(

        FanOut(2),
        stax.parallel(Main, Identity),
        FanInSum,
        Relu
    )

def ConvBlock(kernel_size, filters, strides = (1, 1)):

    kernel_size_ = kernel_size
    # Set the number of convolutional kernels
    filters1, filters2, filters3 = filters

    Main = stax.serial(

        Conv(filters1, (1, 1), strides, padding = "SAME"),
        BatchNorm(),
        Relu,

        Conv(filters2, (kernel_size_, kernel_size_), padding = "SAME"),
        BatchNorm(),
        Relu,

        Conv(filters3, (1, 1), padding = "SAME"),
        BatchNorm()
    )

    Shortcut = stax.serial(
        Conv(filters3, (1, 1), strides, padding = "SAME"),
        BatchNorm()
    )

    return stax.serial(
        FanOut(2), stax.parallel(Main, Shortcut),
        FanInSum,
        Relu)

def ResNet50(number_classes):

    return stax.serial(

        Conv(64, (3, 3), padding = "SAME"),
        BatchNorm(),
        Relu,
        MaxPool((3, 3), strides = (2, 2)),

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

        AvgPool((7, 7)),
        Flatten,
        Dense(number_classes),

        LogSoftmax
    )

