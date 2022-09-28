import jax
import jax.scipy as jsp

def convolve():

    key = jax.random.PRNGKey(17)
    image = jax.random.normal(key, shape = (128, 128, 3))

    kernel_2d = jax.numpy.array([
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]
    ])

    smooth_image = jsp.signal.convolve(image, kernel_2d, mode = "same")

    print("image shape = ", image.shape)
    print("kernel_2d shape = ", kernel_2d.shape)
    print(smooth_image)

def main():

    convolve()

if __name__ == "__main__":

    main()
