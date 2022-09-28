import jax

def convolve():

    """"

    batch size (N), width (W), height (H), channel size (C), kernel input size (I), kernel output size (O)

    """
    image = jax.numpy.zeros((1, 200, 200, 3), dtype = jax.numpy.float32)            # [batch size, height, width, channel size]
    kernel = jax.numpy.zeros((3, 3, 3, 10), dtype = jax.numpy.float32)            # [height, width, kernel input size, kernel output size]

    dimension_numbers = jax.lax.conv_dimension_numbers(image.shape,                 # Shape of input image
                                                       kernel.shape,                # Shape of output image
                                                       ("NHWC", "HWIO", "NHWC"))    # Define the dimensions of input and output

    out = jax.lax.conv_general_dilated(image, kernel, window_strides = [2, 2], padding = "SAME", dimension_numbers = dimension_numbers)

    print("Out shape = ", out.shape)

    dimension_numbers = jax.lax.conv_dimension_numbers(image.shape,
                                                       kernel.shape,
                                                       ("NHWC", "HWIO", "NCHW"))

    out = jax.lax.conv_general_dilated(image, kernel, window_strides = [2, 2], padding = "SAME", dimension_numbers = dimension_numbers)

    print("Out shape = ", out.shape)

def main():

    convolve()

if __name__ == "__main__":

    main()
