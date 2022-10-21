import sys

import jax.numpy
from jax.example_libraries import stax, optimizers
from jax.example_libraries.stax import Conv, Relu, MaxPool, Flatten, Dense, LogSoftmax

sys.path.append("../12.3.1/")
from CharactersConvolution import setup

def CharactersConvolutionalNeuralNetworks(number_classes):

    return stax.serial(

        Conv(1, (3, 3)),
        Relu,

        Conv(1, (5, 5)),
        Relu,

        MaxPool((3, 3), (1, 1)),

        Conv(1, (3, 3)),
        Relu,

        Flatten,

        Dense(256),
        Relu,

        Dense(number_classes),
        LogSoftmax
    )

def verify_accuracy(parameters, batch, predict):

    inputs, targets = batch
    result = predict(parameters, inputs)
    predicted_class = jax.numpy.argmax(result, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)

    return jax.numpy.sum(predicted_class == targets)\

def loss_function(parameters, batch, predict):

    inputs, targets = batch

    losses = -targets * predict(parameters, inputs)
    losses = jax.numpy.sum(losses)

    return jax.numpy.mean(losses)

def update(i, opt_state, batch, get_parameters, opt_update, predict):

    """

    Single optimization step over a minibatch

    """

    parameters = get_parameters(opt_state)
    grad_loss_fucntion = jax.grad(loss_function)
    gradient = grad_loss_fucntion(parameters, batch, predict)

    return opt_update(i, gradient, opt_state)

def initialize():

    trains, labels = setup()

    key = jax.random.PRNGKey(17)

    train_texts = jax.random.permutation(key, trains, independent = True)
    train_labels = jax.random.permutation(key, labels, independent = True)

    test_texts = train_texts[: 12000]
    test_labels = train_labels[: 12000]

    train_texts = train_texts[12000:]
    train_labels = train_labels[12000:]

    input_shape = [-1, 64, 28, 1]

    return key, input_shape, train_texts, train_labels, test_texts, test_labels

def train():

    key, input_shape, train_texts, train_labels, test_texts, test_labels = initialize()

    init_random_parameters, predict = CharactersConvolutionalNeuralNetworks(5)

    # step_size is the learning_rate
    opt_init, opt_update, get_parameters = optimizers.adam(step_size = 2.17e-4)
    _, init_parameters = init_random_parameters(key, input_shape)
    opt_state = opt_init(init_parameters)

    batch_size = 128
    total_batch = 120000 - 12000

    for _ in range(170):

        epochs = int(total_batch / batch_size)

        print(f"Iteration {_} is started")

        for i in range(epochs):

            start = i * batch_size
            end = (i + 1) * batch_size

            batch = train_texts[start: end]
            targets = train_labels[start: end]

            opt_state = update(i, opt_state, (batch, targets), get_parameters, opt_update, predict)

            if (i + 1) % 100 == 0:

                parameters = get_parameters(opt_state)
                loss = loss_function(parameters, (batch, targets), predict)

                print(f"Loss: {loss}")

        parameters = get_parameters(opt_state)

        print(f"Iteration {_} is done")

        accuracys = []
        corrects = 0.0

        test_epochs = int(12000 // batch_size)

        for i in range(test_epochs):

            start = i * batch_size
            end = (i + 1) * batch_size

            batch = test_texts[start: end]
            targets = test_labels[start: end]

            corrects += verify_accuracy(parameters, (batch, targets))

        accuracys.append(corrects / float(total_batch))

        print(f"Training set accuracy: {accuracys}")

if __name__ == "__main__":

    train()
