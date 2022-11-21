import jax.numpy


def compute_convolutional_generaa_dilated():

    # Image shape = [N, H, W, C]
    image = jax.numpy.zeros(shape = (1, 200, 200, 3), dtype = jax.numpy.float32)

    # Kernel shape = [H, W, I, O]
    kernel = jax.numpy.zeros(shape = (3, 3, 3, 10), dtype = jax.numpy.float32)

    # Input shape/Output shape
    dimension_numbers = jax.lax.conv_dimension_numbers(image.shape, kernel.shape, ("NHWC", "HWIO", "NHWC"))

    outputs = jax.lax.conv_general_dilated(lhs = image, rhs = kernel, window_strides = [2, 2], padding = "SAME", dimension_numbers = dimension_numbers)

    print("Convolutional General Dilated Outputs Shape = ", outputs.shape)

    print("--------------------------")

    # Input shape/Output shape
    dimension_numbers = jax.lax.conv_dimension_numbers(image.shape, kernel.shape, ("NHWC", "HWIO", "NCHW"))

    outputs = jax.lax.conv_general_dilated(lhs = image, rhs = kernel, window_strides = [2, 2], padding = "SAME", dimension_numbers = dimension_numbers)

    print("Convolutional General Dilated Outputs Shape = ", outputs.shape)

if __name__ == '__main__':

    compute_convolutional_generaa_dilated()
