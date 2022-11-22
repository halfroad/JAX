import time

import jax
import tensorflow_datasets as tfds


def setup():

    (trains, tests), info = tfds.load(name = "mnist", with_info = True, batch_size = -1, split = [tfds.Split.TRAIN, tfds.Split.TEST], data_dir = "../../../../../../../Shares/Datasets/MNIST/")

    # Extract the informative features
    class_names = info.features["label"].names
    classes = info.features["label"].num_classes

    trains = tfds.as_numpy(trains)
    tests = tfds.as_numpy(tests)

    train_images, train_labels = trains["image"], trains["label"]
    test_images, test_labels = tests["image"], tests["label"]

    return (class_names, classes), ((train_images, train_labels), (test_images, test_labels))

def init_mlp_params(key, kernel_shapes):

    params = []

    # Create kernel for 12 layers of Convolutional Neural Network
    for i in range(len(kernel_shapes) - 2):

        kernel_weight = jax.random.normal(key, shape = kernel_shapes[i]) / jax.numpy.sqrt(784)
        _dict = dict(weight = kernel_weight)

        params.append(_dict)

    # Create kernel for 3 layers of Dense
    for i in range(len(kernel_shapes) - 2, len(kernel_shapes)):

        weight = jax.random.normal(key, shape = kernel_shapes[i]) / jax.numpy.sqrt(784)
        bias = jax.random.normal(key, shape = (kernel_shapes[i][-1],))

        _dict = dict(weight = weight, bias = bias)

        params.append(_dict)

    return params

def one_hot_no_jit(inputs, k = 10, dtype = jax.numpy.float32):

    matches = jax.numpy.array(inputs[:, None] == jax.numpy.arange(k), dtype = dtype)

    return matches

def batch_normalization(inputs, gamma = 0.9, beta = 0.25, epsilon = 1e-9):

    u = jax.numpy.mean(inputs, axis = 0)
    standard_variance = jax.numpy.sqrt(inputs.var(axis = 0) + epsilon)
    y = (inputs - u) / standard_variance
    scale_shift = y * gamma + beta

    return scale_shift

def conv(inputs, weights, window_strides = 1):

    shape = inputs.shape
    dimension_numbers = jax.lax.conv_dimension_numbers(lhs_shape = shape, rhs_shape = weights["weight"].shape, dimension_numbers = ("NHWC", "HWIO", "NHWC"))

    inputs = jax.lax.conv_general_dilated(inputs, weights["weight"], window_strides = [window_strides, window_strides], padding = "SAME", dimension_numbers = dimension_numbers)
    inputs = jax.nn.relu(inputs)

    return inputs

@jax.jit
def forward(params, inputs):

    for i in range(len(params) - 2):

        inputs = conv(inputs, weights = params[i])

    # Replace the pooling layer with batch_normalization layer
    inputs = batch_normalization(inputs)
    inputs = jax.numpy.reshape(inputs, [-1, inputs.shape[0]])

    for i in range(len(params) - 2, len(params) - 1):

        inputs = jax.numpy.matmul(inputs, params[i]["weight"]) + params[i]["bias"]
        inputs = jax.nn.selu(inputs)

    inputs = jax.numpy.matmul(inputs, params[-1]["weight"]) + params[-1]["bias"]
    inputs = jax.nn.softmax(inputs, axis = -1)

    return inputs

@jax.jit
def cross_entropy(genuines, predictions):

    entropys = genuines * jax.numpy.log(jax.numpy.clip(predictions, 1e-9, 0.999)) + (1 - genuines) * jax.numpy.log(jax.numpy.clip(1 - predictions, 1e-9, 0.999))
    entropys = jax.numpy.sum(entropys, axis = 1)
    entropys = jax.numpy.mean(entropys)

    return entropys

@jax.jit
def loss_function(params, inputs, genuines):

    predictions = forward(params, inputs)

    return cross_entropy(genuines, predictions)

@jax.jit
def optimzier(params, inputs, genuines, learn_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)
    params = jax.tree_util.tree_map(lambda param, gradient: param - gradient * learn_rate, params, gradients)

    return params

@jax.jit
def verrify_accuracy(params, inputs, targets):

    """

    Correct the predictions over a mini batch

    """

    # Here is an adaption, the output target is adapted to [-1, 10] because the result of predictions is [-1, 10]
    # There should be a conversion between the 2 jax.numpy.argmax
    result = forward(params, inputs)
    class_ = jax.numpy.argmax(result, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)

    matches = jax.numpy.sum(class_ == targets)

    return matches

def train():

    (class_names, classes), ((train_images, train_labels), (test_images, test_labels)) = setup()

    train_images = jax.numpy.float32(train_images)
    train_labels = one_hot_no_jit(train_labels)

    key = jax.random.PRNGKey(15)

    kernel_shapes = [

        [3, 3, 1, 6],
        [3, 3, 16, 32],
        [3, 3, 32, 48],
        [3, 3, 48, 64],
        [60000, 128],
        [28, 10]
    ]
    params = init_mlp_params(key, kernel_shapes)

    start = time.time()
    batch_size = 500

    for i in range(20):

        batch_number = train_images.shape[0] // batch_size

        for j in range(batch_number):

            start_ = batch_size * j
            end_ = batch_size * (j + 1)

            images_batch = train_images[start_: end_]
            labels_batch = train_labels[start_: end_]

            params = optimzier(params, images_batch, labels_batch)

        if (i + 1) % 100 == 0:

            loss = loss_function(params, train_images, train_labels)
            end = time.time()

            accuracies = verrify_accuracy(params, test_images, test_labels) / float(4096.)

            print(f"Now the loss is {loss} after {i + 1} epochs in {end - start}s, the accracy is {accuracies}")

            start = time.time()


def test():

    train()

if __name__ == '__main__':

    test()
