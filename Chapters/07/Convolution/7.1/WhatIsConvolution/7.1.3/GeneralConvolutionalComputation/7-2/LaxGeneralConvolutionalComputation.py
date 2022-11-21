import jax


def lax_convolve():

    image = jax.numpy.zeros(shape = (1, 200, 198, 3), dtype = jax.numpy.float32)
    kernel = jax.numpy.zeros(shape = (10, 3, 3, 3), dtype = jax.numpy.float32)

    print("Image.shape = ", image.shape)
    print("Kernel.shape = ", kernel.shape)

    transposed_image = jax.numpy.transpose(image, [0, 3, 1, 2])

    print("transposed_image.shape = ", transposed_image.shape)

    outputs = jax.lax.conv(lhs = transposed_image, rhs = kernel, window_strides = [1, 1], padding = "SAME")

    print("Outputs shape = ", outputs.shape)

if __name__ == '__main__':

    lax_convolve()

