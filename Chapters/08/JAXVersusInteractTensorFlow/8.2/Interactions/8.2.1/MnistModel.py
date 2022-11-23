import time

import jax
import tensorflow_datasets as tfds

def setup():

    # Download the mnist dataset from tensorflow-dataset directly
    trains, tests = tfds.load(name = "mnist", split = [tfds.Split.TRAIN, tfds.Split.TRAIN], batch_size = -1, data_dir = "../../../../../../Shares/Datasets/MNIST/")

    tests = tfds.as_numpy(tests)
    trains = tfds.as_numpy(trains)

    train_images, train_labels = trains["image"], trains["label"]
    test_images, test_labels = tests["image"], tests["label"]

    _, height, width, channels = train_images.shape

    # Get the dimension of dataset
    pixels = height * width * channels

    # Set the number of classifications
    output_dimensions = 10

    # Adapt the inputs dimension
    train_images = jax.numpy.reshape(train_images, (-1, pixels))
    test_images = jax.numpy.reshape(test_images, (-1, pixels))

    # One Hot the labels
    train_labels = one_hot(inputs = train_labels, k = output_dimensions)
    test_labels = one_hot(inputs = train_labels, k = output_dimensions)

    key = jax.random.PRNGKey(15)

    return (key, pixels, output_dimensions), (train_images, train_labels), (test_images, test_labels)

def one_hot(inputs, k = 10, dtype = jax.numpy.float32):

    matches = jax.numpy.array(inputs[:, None] == jax.numpy.arange(k), dtype)

    return matches

def init_mlp_params(key, pixels, output_dimensions):

    params = []

    layer_dimensions = [pixels, 512, 256, output_dimensions]

    for i in range(1, len(layer_dimensions)):

        weight = jax.random.normal(key, shape = (layer_dimensions[i - 1], layer_dimensions[i])) / jax.numpy.sqrt(pixels)
        bias = jax.random.normal(key, shape = (layer_dimensions[i],)) / jax.numpy.sqrt(pixels)

        _dict = {"weight": weight, "bias": bias}

        params.append(_dict)

    return params

# Prediction
def forward(params, inputs):

    for param in params[: -1]:

        weight = param["weight"]
        bias = param["bias"]

        inputs = jax.numpy.dot(inputs, weight) + bias
        inputs = relu(inputs)

    outputs = jax.numpy.dot(inputs, params[-1]["weight"]) + params[-1]["bias"]

    print(outputs.shape)

    outputs = jax.nn.softmax(outputs, axis = -1)

    return outputs

import jax.numpy

@jax.jit
def relu(inputs):

    return jax.numpy.maximum(0, inputs)

@jax.jit
def cross_entropy(genuines, predictions):

    entropys = genuines * jax.numpy.log(jax.numpy.clip(predictions, 1e-9, 0.999)) + (1 - genuines) * jax.numpy.log(jax.numpy.clip(1 - predictions, 1e-9, 0.999))
    entropys = -jax.numpy.sum(entropys)

    return jax.numpy.mean(entropys)

@jax.jit
def forward(params, inputs):

    for param in params[: -1]:

        weight = param["weight"]
        bias = param["bias"]

        inputs = jax.numpy.dot(inputs, weight) + bias
        inputs = relu(inputs)

    outputs = jax.numpy.dot(inputs, params[-1]["weight"]) + params[-1]["bias"]

    print(outputs.shape)

    outputs = jax.nn.softmax(outputs, axis = -1)

    return outputs

@jax.jit
def loss_function(params, inputs, genuines):

    predictions = forward(params, inputs)

    return cross_entropy(genuines, predictions)

@jax.jit
def optimizer(params, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)

    params = jax.tree_util.tree_map(lambda param, gradient: param - learning_rate * gradient, params, gradients)

    return params

@jax.jit
def verify_accuracy(params, inputs, targets):

    result = forward(params, inputs)
    class_ = jax.numpy.argmax(result, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)

    equals = jax.numpy.sum(class_ == targets)

    return equals

def train():

    (key, pixels, output_dimensions), (train_images, train_labels), (test_images, test_labels) = setup()

    params = init_mlp_params(key, pixels, output_dimensions)
    start = time.time()

    for i in range(500):

        params = optimizer(params, train_images, train_labels)

        if (i + 1) % 50 == 0:

            loss = loss_function(params, test_images, test_labels)
            accuracy = verify_accuracy(params, test_images, test_labels) / float(10000.0)

            end = time.time()

            print(f"%.12fs" % (end - start), f"is consumed while iterating {i + 1} epochs, the loss now is {loss}, the accuracy is {accuracy}")

            start = time.time()

if __name__ == '__main__':

    train()
