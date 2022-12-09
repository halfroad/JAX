import csv
import re
import ssl

import jax
import nltk
import numpy
from gensim.parsing import PorterStemmer
from jax.example_libraries import stax, optimizers
from nltk.corpus import stopwords


def text_purify(string: str, stops):

    # Lower the string
    string = string.lower()

    # Replace the nonstandard alphabat, includes punctuation makes
    string = re.sub(r"[^a-z]", repl = " ", string = string)

    # Replace multiple spaces
    string = re.sub(r" +", repl = " ", string = string)

    # Trim the spaces in head and tail
    string = string.strip("")

    # Split the string with Space
    strings = string.split(" ")

    # clean the stop words
    strings = [word for word in strings if word not in stops]

    # Restore the word root meaning
    strings = [PorterStemmer().stem(word) for word in strings]

    # Add the prefix of start
    strings = ["bos"] + strings

    # Add the suffix of end
    strings = strings + ["eos"]

    return strings

def title_purify(string: str):

    # Lower the string
    string = string.lower()

    # Replace the nonstandard alphabat, includes punctuation makes
    string = re.sub(r"[^a-z]", repl = " ", string = string)

    # Replace multiple spaces
    string = re.sub(r" +", repl = " ", string = string)

    # Trim the spaces in head and tail
    string = string.strip()
    # string = re.sub(" ", "", string)

    # Add the suffix of end
    string = string + " eos"

    return string

def one_hot(alphabet: str, characters):

    array = jax.numpy.array(characters)

    length = len(alphabet) + 1
    # Diagonal Points are 1, others are 0
    matrix = jax.numpy.eye(length)[array]

    return matrix

def numberic_one_hot(numbers):

    array = numpy.array(numbers)
    maximum = jax.numpy.max(array) + 1

    matrix = jax.numpy.eye(maximum)[array]

    return matrix

def string_to_indexed_numbers(alphabet: str, string: str):

    numbers = []

    for character in string:

        number = alphabet.index(character)
        numbers.append(number)

    return numbers

def padding_numbers_matrix(alphabet: str, string: str, maximum: int = 64):

    length = len(string)

    if length > 64:

        string = string[: 64]

        numbers = string_to_indexed_numbers(alphabet, string)
        matrix = one_hot(alphabet, numbers)

        return matrix

    else:

        numbers = string_to_indexed_numbers(alphabet, string)
        matrix = one_hot(alphabet, numbers)

        # Padding length
        length = maximum - length

        # Padding with 0
        padding = jax.numpy.zeros([length, 28])

        # Concatenanate the characters matrix with padded_matrix
        matrix = jax.numpy.concatenate([matrix, padding])

        return matrix

def download_stop_words(path):

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("stopwords", download_dir = path)

    nltk.data.path.append(path)

def stop_words():

    root = "../../../../../../Shares/Datasets//NLTK/"
    path = root + "stopwords"

    download_stop_words(path)

    stops = stopwords.words("english")

    return stops

def prepare():

    def split(trains_):

        labels = []
        titles = []
        descriptions = []

        stops = stop_words()

        for index, line in enumerate(trains_):

            if index > 0:

                labels.append(jax.numpy.int32(line[0]))

                title = title_purify(line[1].lower())

                titles.append(title)
                descriptions.append(text_purify(line[2].lower(), stops = stops))

        return labels, titles, descriptions

    def make_datasets(labels, titles, descriptions):

        alphabet = "abcdefghijklmnopqrstuvwxyz "

        trains_ = []
        labels = numberic_one_hot(labels)

        for title in titles:

            matrix = padding_numbers_matrix(alphabet, title)

            trains_.append(matrix)

        labels, trains_, descriptions = numpy.array(labels), numpy.array(trains_), numpy.array(descriptions)

        trains_ = jax.numpy.expand_dims(trains_, axis = -1)

        return labels, trains_, descriptions

    handle = open("../../../../../../Shares/Datasets/agnews/train.csv", mode = "r")
    trains = csv.reader(handle)

    labels_, titles_, descriptions_ = split(trains)

    return make_datasets(labels_, titles_, descriptions_)

def CharacterConvolutionalNeuronNetwork(num_classes):

    return stax.serial(

        stax.Conv(1, (3, 3)),
        stax.Relu,

        stax.Conv(1, (5, 5)),
        stax.Relu,

        stax.Conv(1, (3, 3)),
        stax.Relu,

        stax.Flatten,

        stax.Dense(256),
        stax.Dense(num_classes),

        stax.LogSoftmax
    )

def verify_accuracy(params, batch, predict):

    inputs, targets = batch
    result = predict(params, inputs)
    class_ = jax.numpy.argmax(targets, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)

    return jax.numpy.sum(class_ == targets)


def loss_function(params, batch, predict):

    inputs, targets = batch

    predictions = predict(params, inputs)

    losses = -targets * predictions
    losses = jax.numpy.sum(losses, axis = 1)

    return jax.numpy.mean(losses)

def update(i, optimizer_state, batch, get_params, optimizer_update):

    """
    Single optimization step over a mini batch
    """

    params = get_params(optimizer_state)
    grad_loss_function = jax.grad(loss_function)

    gradients = grad_loss_function(params, batch)

    params = optimizer_update(i, gradients, optimizer_state)

    return params

def setup():

    input_shape = [-1, 64, 28, 1]
    labels, titles, descriptions = prepare()
    key = jax.random.PRNGKey(15)

    descriptions = jax.random.shuffle(key, descriptions)

    train_texts = jax.random.shuffle(key, titles)
    train_labels = jax.random.shuffle(key, labels)

    train_texts = train_texts[12000:]
    train_labels = train_labels[12000:]

    test_texts = train_texts[: 12000]
    test_labels = train_labels[: 12000]

    init_random_params, predict = CharacterConvolutionalNeuronNetwork(5)

    batch_size = 128
    total_number = 120000 - 12000

    return ((train_texts, train_labels), (test_texts, test_labels)), (input_shape, key, batch_size, total_number), (init_random_params, predict)

def train():

    ((train_texts, train_labels), (test_texts, test_labels)), (input_shape, key, batch_size, total_number), (init_random_params, predict) = setup()

    optimizer_init, optimizer_update, get_params = optimizers.adam(step_size = 2.17e-4)
    _, init_params = init_random_params(key, input_shape = input_shape)

    epochs = int(total_number // batch_size)

    for i in range(epochs):

        print(f"Training epoch {i + 1} started")

        start = i * batch_size
        end = (i + 1) * batch_size

        texts = train_texts[start: end]
        labels = train_labels[start: end]

        optimizer_state = update(i, optimizer_state, (texts, labels))

        if (i + 1) % 100 == 0:

            params = get_params(optimizer_state)
            loss = loss_function(params, (texts, labels))

            print(f"Loss = {loss}")

        params = get_params(optimizer_state)

        print("Training epoch {i + 1} completed")

    accuracies = []
    correct_predictions = 0.0

    test_epochs = int(12000 // batch_size)

    for i in range(test_epochs):

        start = i * batch_size
        end = (i + 1) * batch_size

        texts = train_texts[start: end]
        labels = train_labels[start: end]

        correct_predictions += verify_accuracy(params, (texts, labels))

    accuracies.append(correct_predictions / float(total_number))

    print(f"Training set accuracy: {accuracies}")

if __name__ == '__main__':

    train()





