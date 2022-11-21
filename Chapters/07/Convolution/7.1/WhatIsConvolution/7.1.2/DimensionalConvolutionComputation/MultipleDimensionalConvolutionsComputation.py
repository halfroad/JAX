import jax


def compute_multiple_dimensional_convolution():

    key = jax.random.PRNGKey(15)

    image = jax.random.normal(key, shape = (128, 128, 3))

    print("Image = ", image)

    kernel_2d = jax.numpy.array([[[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]]])

    print("kernel_2d = ", kernel_2d)

    smooth_image = jax.scipy.signal.convolve(image, kernel_2d, mode = "same")

    print("Smooth image = ", smooth_image)

if __name__ == '__main__':

    compute_multiple_dimensional_convolution()
