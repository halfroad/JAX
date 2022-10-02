import sys
import time

import jax.random

sys.path.append("../../7.2.2/")
sys.path.append("../")
import BatchNormalization
import MnistDatasetsV1
import ConvolutionUtils

def init_multilayers_perceptron_parameters(shapes):

    parameters = []

    key = jax.random.PRNGKey(17)

    # Create the kernel utilized by 12 layers Convolutional Neural Networks
    for i in range(len(shapes) - 2):

        weight = jax.random.normal(key, shape = shapes[i]) / jax.numpy.sqrt(28. * 28.)
        _dict = dict(weight = weight)

        parameters.append(_dict)

    # Create the kernel utilized by 3 layers Dense
    for i in range(len(shapes) - 2, len(shapes)):

        weight = jax.random.normal(key, shape = shapes[i]) / jax.numpy.sqrt(28. * 28.)
        bias = jax.random.normal(key, shape = (shapes[i][-1],)) / jax.numpy.sqrt(28. * 28.)

        _dict = dict(weight = weight, bias = bias)

        parameters.append(_dict)

    return parameters

@jax.jit
def convolve(inputs, kernel, strides = 1):

    weight = kernel["weight"]

    """
    inputs.shape = [N, H, W, C]. lernel.shape = [H, W, I, O]
    out.shape = [N, H, W, C]
    ConvDimensionNumbers(lhs_spec=(0, 3, 1, 2), rhs_spec=(3, 2, 0, 1), out_spec=(0, 3, 1, 2))
    """
    dimension_numbers = jax.lax.conv_dimension_numbers(inputs.shape, weight.shape, ("NHWC", "HWIO", "NHWC"))

    inputs = jax.lax.conv_general_dilated(inputs, weight, window_strides = [strides, strides], padding = "SAME", dimension_numbers = dimension_numbers)
    inputs = jax.nn.selu(inputs)

    return inputs

@jax.jit
def forward(parameters, inputs):

    for i in range(len(parameters) - 2):

        inputs = convolve(inputs, kernel = parameters[i])

    inputs = BatchNormalization.batch_normalize(inputs)
    inputs = jax.numpy.reshape(inputs, [inputs.shape[0], -1])

    for i in range(len(parameters) - 2, len(parameters) - 1):

        inputs = jax.numpy.matmul(inputs, parameters[i]["weight"]) + parameters[i]["bias"]
        inputs = jax.nn.selu(inputs)

    inputs = jax.numpy.matmul(inputs, parameters[-1]["weight"]) + parameters[-1]["bias"]
    inputs = jax.nn.softmax(inputs, axis = -1)

    return inputs

@jax.jit
def cross_entropy(genuines, predictions):

    entropys = genuines * jax.numpy.log(jax.numpy.clip(predictions, 1e-9, .999)) + (1 - genuines) * jax.numpy.log(jax.numpy.clip(1 - predictions, 1e-9, .999))
    entropys = -jax.numpy.sum(entropys, axis = 1)

    return jax.numpy.mean(entropys)

@jax.jit
def loss_function(paramters, inputs, genuines):

    predictions = forward(paramters, inputs)
    entropys = cross_entropy(genuines, predictions)

    return entropys

@jax.jit
def optimizer_function(paramters, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(paramters, inputs, genuines)

    new_paramters = jax.tree_util.tree_map(lambda parameter, gradient: parameter - learning_rate * gradient, paramters, gradients)

    return new_paramters

@jax.jit
def prediction_correct(parameters, inputs, targets):

    """

    Correct the predictions over a mini batch

    """

    # Here is an adaption. Since the shape of prediction is [-1, 10], so the target is adjusted to [-1, 10]
    # Conversion shall be conducted between 2 jax.numpy.argmax
    predictions = forward(parameters, inputs)
    classification = jax.numpy.argmax(predictions, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)

    return jax.numpy.sum(classification == targets)

def setup():

    train_images, train_labels, test_images, test_labels = MnistDatasetsV1.mnist()

    batch_size = 312
    inputs_channels = 1
    # Be noted that the height (H) and width (W) means the height and width of array.
    inputs_shape = [1, 28, 28, inputs_channels]                 # shape = [N, H, W, C]
    kernel_shape = [3, 3, inputs_channels, inputs_channels]     # shape = [H, W, I, O]

    return (train_images, train_labels), (test_images, test_labels), (batch_size, inputs_shape, kernel_shape)

def train():

    (train_images, train_labels), (test_images, test_labels), (batch_size, inputs_shape, kernel_shape) = setup()

    train_images = ConvolutionUtils.partial_flatten(train_images)
    test_images = ConvolutionUtils.partial_flatten(test_images)
    train_labels = ConvolutionUtils.one_hot_nojit(train_labels)
    test_labels = ConvolutionUtils.one_hot_nojit(test_labels)

    print(f"train_images.shape = {train_images.shape}, train_labels.shape = {train_labels.shape}), (test_images.shape = {test_images.shape}, test_labels.shape = {test_labels.shape}")

    """
    
    train_images = train_images[: 2000]
    train_labels = train_labels[: 2000]

    test_images = test_images[: 100]
    test_labels = test_labels[: 100]
    
    
    train_images.shape = (60000, 28, 28, 1),
    train_labels.shape = (60000, 10),
    test_images.shape = (10000, 28, 28, 1),
    test_labels.shape = (10000, 10)
    
    """
    kernel_shapes = [
        [3, 3, 1, 16],
        [3, 3, 16, 32],
        [3, 3, 32, 48],
        [3, 3, 48, 64],
        [50176, 128],
        [128, 10]
    ]

    parameters = init_multilayers_perceptron_parameters(kernel_shapes)

    begin = time.time()

    for i in range(20):

        batch_number = train_images.shape[0] // batch_size

        for j in range(batch_number):

            start_ = batch_size * j
            end_ = batch_size * (j + 1)

            train_images_batch = train_images[start_: end_]
            train_labels_batch = train_labels[start_: end_]

            parameters = optimizer_function(parameters, train_images_batch, train_labels_batch)

        if (i + 1) % 5 == 0:

            loss = loss_function(parameters, train_images, train_labels)

            end = time.time()

            accuracy = prediction_correct(parameters, test_images, test_labels) / float(4096.)

            print(f"With {i +1} epoches, now the loss is {loss}, the accuracy of test set is {accuracy}")

            begin = time.time()
def main():

    train()

if __name__ == "__main__":

    main()
