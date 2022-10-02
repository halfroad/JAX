import time
import jax
import tensorflow_datasets as tfds

def setup():

    # Load the datasets from tensorflow-datasets
    trains = tfds.load("mnist", split = tfds.Split.TRAIN, batch_size = -1)
    trains = tfds.as_numpy(trains)

    train_images, train_labels, = trains["image"], trains["label"]

    _, height, width, channels = train_images.shape

    # Read the dimensions
    number_pixels = height * width * channels

    # Set the number of classifications
    output_dimensions = 10

    # Change the dimensions of input dataset
    train_images = jax.numpy.reshape(train_images, (-1, number_pixels))

    # One-Hot
    train_labels = one_hot(train_labels, k = output_dimensions)

    tests = tfds.load("mnist", split = tfds.Split.TEST, batch_size = -1)
    tests = tfds.as_numpy(tests)

    test_images, test_labels = tests["image"], tests["label"]

    test_images = jax.numpy.reshape(test_images, (-1, number_pixels))
    test_labels = one_hot(test_labels, k = output_dimensions)

    return (train_images, train_labels), (test_images, test_labels), number_pixels, output_dimensions

def init_parameters(number_pixel, output_dimensions):

    def init(layer_dimensions = [number_pixel, 512, 256, output_dimensions]):

        parameters = []

        key = jax.random.PRNGKey(17)

        for i in range(1, (len(layer_dimensions))):

            weight = jax.random.normal(key, shape = (layer_dimensions[i - 1], layer_dimensions[i])) / jax.numpy.sqrt(number_pixel)
            bias = jax.random.normal(key, shape = (layer_dimensions[i],)) / jax.numpy.sqrt(number_pixel)

            _dict = {"weight": weight, "bias": bias}

            parameters.append(_dict)

        return parameters

    return init()

# One-Hot encoding the labels
def one_hot(inputs, k = 10, dtype = jax.numpy.float32):

    """

    Create a one-hot encoding of inputs of size k.

    """

    return jax.numpy.array(inputs[:, None] == jax.numpy.arange(k), dtype)

def forward(parameters, inputs):

    for parameter in parameters[: -1]:

        weight = parameter["weight"]
        bias = parameter["bias"]

        inputs = jax.numpy.dot(inputs, weight) + bias
        inputs = relu(inputs)

    output = jax.numpy.dot(inputs, parameters[-1]["weight"]) + parameters[-1]["bias"]
    output = jax.nn.softmax(output, axis = -1)

    return output

@jax.jit
def relu(input_):

    """

    Activation function

    """

    return jax.numpy.maximum(0, input_)

@jax.jit
def cross_entropy(genuines, predictions):

    """

    Cross-entropy function

    """

    entropys = genuines * jax.numpy.log(jax.numpy.clip(predictions, 1e-9, .999)) + (1 - genuines) * jax.numpy.log(jax.numpy.clip(1 - predictions, 1e-9, .999))
    entropys = -jax.numpy.sum(entropys, axis = 1)

    return jax.numpy.mean(entropys)

@jax.jit
def loss_function(parameters, inputs, genuines):

    """

    Loss function

    """

    # predictions = forward(parameters, inputs)
    # entropys = cross_entropy(genuines, predictions)

    # Appendix 5
    vmap_forward = jax.vmap(forward, [None, 0])
    predictions = vmap_forward(parameters, inputs)
    entropys = cross_entropy(genuines, predictions)

    return entropys

@jax.jit
def optimizer(parameters, inputs, genuines, learning_rate = 1e-3):

    """

    SGD optimizer function

    """
    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(parameters, inputs, genuines)

    parameters = jax.tree_util.tree_map(lambda parameter, gradient: parameter - gradient * learning_rate, parameters, gradients)

    return parameters

@jax.jit
def accuracy_checkup(parameters, inputs, targets):

    """

    Calculus of accuracy

    """

    result = forward(parameters, inputs)
    classification = jax.numpy.argmax(result, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)
    equals = jax.numpy.sum(classification == targets)

    return jax.numpy.sum(equals)

def train(train_images, train_labels, test_images, test_labels, number_pixels, output_dimensions):

    parameters = init_parameters(number_pixels, output_dimensions)

    begin = time.time()

    for i in range(500):

        parameters = optimizer(parameters, train_images, train_labels)

        if (i + 1) % 50 == 0:

            loss = loss_function(parameters, test_images, test_labels)
            end = time.time()
            accuracy = accuracy_checkup(parameters, test_images, test_labels) / float(10000.)

            print("%.12fs is consumed" % (end - begin), f"while iterating {i + 1}, the loss now is %.12f," % loss, "the accuracy of test dataset is %.12f" % accuracy)

            begin = time.time()

def main():

    (train_images, train_labels), (test_images, test_labels), number_pixels, output_dimensions = setup()

    train(train_images, train_labels, test_images, test_labels, number_pixels, output_dimensions)

if __name__ == "__main__":

    main()
