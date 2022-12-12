import csv
import re
import jax
import numpy
from gensim.models import word2vec


def text_purify(string: str):

    string = string.lower()
    string = re.sub(r"[^a-z]", " ", string)
    string = re.sub(r" +", " ", string)
    string = string.strip()
    string = string + " eos"

    strings = string.split(" ")

    return strings

def numberic_one_hot(numbers):

    array = jax.numpy.array(numbers)
    maximum = jax.numpy.max(array) + 1

    matrix = jax.numpy.eye(maximum)[array]

    return matrix

def setup():

    def split(dataset):

        labels = []
        titles = []

        for index, line in enumerate(dataset):

            if index > 0:

                labels.append(jax.numpy.int32(line[0]))

                title = text_purify(line[1])
                titles.append(title)

        return titles, labels

    def make_dataset(titles, targets, maximum = 12):

        model = word2vec.Word2Vec(titles, vector_size = 64, min_count = 1, window = 5)
        matrixes = []

        for line in titles:

            length = len(line)

            if length > maximum:

                line = line[: maximum]
                matrix = model.wv[line]

                matrixes.append(matrix)

            else:

                matrix = model.wv[line]
                padding_length = maximum - length

                # Create a padding matrix
                padding_matrix = jax.numpy.zeros([padding_length, 64]) + 1e-10
                matrix = jax.numpy.concatenate([matrix, padding_matrix])

                matrixes.append(matrix)

        matrixes = numpy.expand_dims(matrixes, 3)
        targets = numberic_one_hot(targets)

        return matrixes, targets

    handle = open("../../../../../../Shares/Datasets/agnews/train.csv", mode = "r")
    trains = csv.reader(handle)

    titles, labels = split(trains)

    return make_dataset(titles, labels)

def main():

    trains, labels = setup()

    print(trains.shape, labels.shape)

if __name__ == '__main__':

    main()
