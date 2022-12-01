from jax.example_libraries.stax import Conv


def model():

    # Number of convolutional kernels, the data dimensions will be generated after processing
    filter_number = 64
    # Size of convolutional kernel
    filter_size = (3, 3)
    # Step strides
    strides = (2, 2)

    Conv(filter_number, filter_size, strides)
    Conv(filter_number, filter_size, strides, padding = "SAME")

def train():

    model()

if __name__ == '__main__':

    train()



