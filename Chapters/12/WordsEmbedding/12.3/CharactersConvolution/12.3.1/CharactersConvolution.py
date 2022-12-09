import csv
import re
import ssl

import jax.numpy
import nltk
import numpy
from gensim.parsing import PorterStemmer
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

def setup():

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

        trains_ = jax.numpy.expand_dims(trains_, axis = -1)

        return jax.numpy.array(labels), jax.numpy.array(trains_), jax.numpy.array(descriptions)

    handle = open("../../../../../../Shares/Datasets/agnews/train.csv", mode = "r")
    trains = csv.reader(handle)

    labels_, titles_, descriptions_ = split(trains)

    return make_datasets(labels_, titles_, descriptions_)

def main():

    labels, titles, descriptions = setup()

    print(labels.shape, titles.shape, descriptions.shape)


if __name__ == '__main__':

    main()
