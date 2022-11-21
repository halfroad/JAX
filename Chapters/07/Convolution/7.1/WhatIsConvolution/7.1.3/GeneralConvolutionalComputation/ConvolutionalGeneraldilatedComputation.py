import jax


def compute_convolutional_general_dilated():

    kernel = jax.numpy.zeros(shape = (10, 3, 3, 3), dtype = jax.numpy.float32)
    image = jax.numpy.zeros(shape = (1, 200, 198, 3), dtype = jax.numpy.float32)

    print("Image.shape = ", image.shape)
    print("Kernel.shape = ", kernel.shape)

    transposed_image = jax.numpy.transpose(image, [0, 3, 1, 2])

    print("transposed_image.shape = ", transposed_image.shape)

    outputs = jax.lax.conv_general_dilated(transposed_image, kernel, window_strides = [2, 2], padding = "SAME")

    print("Outputs shape = ", outputs.shape)

if __name__ == '__main__':

    compute_convolutional_general_dilated()
