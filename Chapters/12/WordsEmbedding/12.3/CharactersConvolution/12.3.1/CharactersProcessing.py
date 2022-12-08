import csv
import re

import jax.numpy
import numpy
from gensim.parsing import PorterStemmer

def setup():

    def split(trains_):

        labels = []
        titles = []
        descriptions = []

        for index, line in enumerate(trains_):

            if index > 0:

                labels.append(jax.numpy.int32(line[0]))

                title = text_purify(line[1].lower())

                titles.append(title)
                descriptions.append(line[2].lower())

        return labels, titles, descriptions

    def make_datasets(labels, titles, descriptions):

        alphabet = "abcdefghijklmnopqrstuvwxyz"

        trains_ = []
        labels = numberic_one_hot(labels)

        for title in titles:

            matrix = padding_numbers_matrix(alphabet, title)

            trains_.append(matrix)

        return labels, trains_, descriptions

    handle = open("../../../../../../Shares/Datasets/agnews/train.csv", mode = "r")
    trains = csv.reader(handle)

    labels_, titles_, descriptions_ = split(trains)

    return make_datasets(labels_, titles_, descriptions_)

def text_purify(string: str):

    # Lowering
    title = string.lower()
    # Replace the non-alphabet, ^ is reversable operator
    string = re.sub(r"[^a-z]", "", string)
    string = re.sub(r" +", " ", string)
    string = re.sub(" ", "", string)

    string = string.strip()
    # string = string + " eos"

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
        padding = jax.numpy.zeros([length, 27])

        # Concatenanate the characters matrix with padded_matrix
        matrix = jax.numpy.concatenate([matrix, padding])

        return matrix

def main():

    labels, titles, descriptions = setup()

    print(labels.shape, titles.shape, descriptions.shape)


if __name__ == '__main__':

    main()
