import jax.lax
import MnistDatasetsV1

import sys
sys.path.append("../7.2.2/")
from BatchNormalization import batch_normalize

def setup():

    train_images, train_labels, test_images, test_labels = MnistDatasetsV1.mnist()

    batch_size = 312
    image_channel_dimension = 1

    return (train_images, train_labels), (test_images, test_labels), (batch_size, image_channel_dimension)

# Convolution
def convolve(inputs, kernel_weight, strides = 1):

    inputs_shape = inputs.shape

    dimension_numbers = jax.lax.conv_dimension_numbers(inputs_shape, kernel_weight["kernel_weight"].shape, ("NHWC", "HWIO", "NHWC"))
    inputs = jax.lax.conv_general_dilated(inputs, kernel_weight["kernel_weight"], window_strides = [strides, strides], padding = "SAME", dimension_numbers = dimension_numbers)
    inputs = jax.nn.selu(inputs)

    return inputs

@jax.jit
def forward(parameters, inputs):

    for i in range(len(parameters) - 2):

        inputs = convolve(inputs, kernel_weight = parameters[i])

    # Replace the pooling layer with normalization
    inputs = batch_normalize(inputs)
    inputs = jax.numpy.reshape(inputs, [-1, 50176])

    for i in range(len(parameters) - 2, len(parameters) - 1):

        inputs = jax.numpy.matmul(inputs, parameters[i]["weight"]) + parameters[i]["bias"]
        inputs = jax.nn.selu(inputs)

    inputs = jax.numpy.matmul(inputs, parameters[-1]["weight"]) + parameters[-1]["bias"]
    inputs = jax.nn.softmax(inputs, axis = -1)

    return inputs

def start():

    (train_images, train_labels), (test_images, test_labels), (batch_size, image_channel_dimension) = setup()

    print(train_images, train_labels)

def main():

    start()

if __name__ == "__main__":

    main()

