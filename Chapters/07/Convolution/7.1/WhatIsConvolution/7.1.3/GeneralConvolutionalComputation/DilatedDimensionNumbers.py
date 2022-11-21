import jax


def dilated_dimension_numbers():

    # Image Shape = [N, H, W, C]
    image = jax.numpy.zeros((1, 200, 200, 3), dtype = jax.numpy.float32)

    # Kernel Shape = [H, W, I, O]
    kernel = jax.numpy.zeros(shape = (3, 3, 3, 10))

    # Define the input/kernel/output shapes
    dimension_numbers = jax.lax.conv_dimension_numbers(image.shape, kernel.shape, dimension_numbers = ("NHWC", "HWIO", "NHWC"))

    print("Dimension Numbers = ", dimension_numbers)

if __name__ == '__main__':

    dilated_dimension_numbers()
