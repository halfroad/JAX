import jax

def convolve():

    #image = jax.numpy.zeros(shape = (1, 3, 200, 198), dtype = jax.numpy.float32)
    image = jax.numpy.zeros(shape = (1, 200, 198, 3), dtype=jax.numpy.float32)

    kernel = jax.numpy.zeros(shape = (10, 3, 3, 3), dtype=jax.numpy.float32)

    print("Image shape = ", image.shape)
    print("Kernel shape = ", kernel.shape)

    transposed = jax.numpy.transpose(image, [0, 3, 1, 2])

    print("transposed shape = ", transposed.shape)

    out = jax.lax.conv(transposed, kernel, window_strides = [1, 1], padding = "SAME")

    print("Out shape = ", out.shape)


def dilation_convolve():

    image = jax.numpy.zeros(shape = (1, 200, 198, 3), dtype = jax.numpy.float32)
    kernel = jax.numpy.zeros(shape = (10, 3, 3, 3), dtype = jax.numpy.float32)

    print("Image shape = ", image.shape)
    print("Kernel shape = ", kernel.shape)

    transposed = jax.numpy.transpose(image, [0, 3, 1, 2])

    print("transposed = ", transposed.shape)

    # out = jax.lax.conv_general_dilated(transposed, kernel, window_strides = [1, 2], padding="SAME")
    # out = jax.lax.conv_general_dilated(transposed, kernel, window_strides = [3, 3], padding="SAME")
    out = jax.lax.conv_general_dilated(transposed, kernel, window_strides = [2, 2], padding="SAME")

    print("Out shape:", out.shape)


def dimension_numbers_convolve():

    """

    batch size, height, width, channel size, kernel input size, kernel output size

    """

    # shape = [batch size, height, width, channel size]
    image = jax.numpy.zeros((1, 200, 200, 3), dtype = jax.numpy.float32)

    # shape = [height, width, kernel input size, kernel outout size]
    kernel = jax.numpy.zeros(shape = (3, 3, 3, 10), dtype = jax.numpy.float32)

    dimension_numbers = jax.lax.conv_dimension_numbers(image.shape,  # Shape (Dimensions) of input image
                                                       kernel.shape,  # Shape (Dimensions) of output image
                                                       ("HNWC", "HWIO", "NHWC"))

    print(dimension_numbers)


def main():

    convolve()

    print("----------------------------------")

    dilation_convolve()

    print("----------------------------------")

    dimension_numbers_convolve()


if __name__ == "__main__":

    main()
