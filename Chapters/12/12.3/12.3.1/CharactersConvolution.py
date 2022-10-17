import re
import sys
import numpy

sys.path.append("../../12.1/12.1.1/12-1/")
sys.path.append("../../12.1/12.1.2/12-5/")

from DatasetsBuilder import read

"""

12.3.1
1st step: The read of titles and convertion

"""
def refine(string: str):

    # Lowering
    title = string.lower()
    # Replace the non-alphabet, ^ is reversable operator
    string = re.sub(r"[^a-z]", " ", string)
    string = re.sub(r" +", " ", string)
    string = re.sub(" ", "", string)

    string = string.strip()

    string = string + " eos"

    return string

"""

12.3.1
2nd step: one-hot

"""
def one_hot(orginal, alphabet = None):

    if alphabet is None:
        alphabet = "abcdefghijklmnopqrstuvwxyz "

    values = numpy.array(orginal)
    number = len(alphabet) + 1

    # Return a 2-D array with ones on the diagonal and zeros elsewhere.
    eyes = numpy.eye(number)[values]

    return eyes

"""

12.3.1
2nd step: Convert the string to indexed numbers

"""

def string_to_indexed_numbers(string: str, alphabet = None):

    if alphabet is None:
        alphabet = "abcdefghijklmnopqrstuvwxyz "

    indexed_numbers = []

    for character in string:

        number = alphabet.index(character)
        indexed_numbers.append(number)

    return indexed_numbers

"""

12.3.1
2nd step: Convert the string to maxtrix

"""
def string_to_matrix(string: str):

    indexed_numbers = string_to_indexed_numbers(string)
    matrix = one_hot(indexed_numbers)

    return matrix

"""

12.3.1
2nd step: Align the trailing empty maxtrix - padding with maximum line

"""
def padding_string_matrix(string, padding_start = 64):

    """

    padding_start: the max length of padding

    """
    length = len(string)

    if length > 64:

        string = string[: 64]
        string_matrix = string_to_matrix(string)

        return string_matrix

    else:

        string_matrix = string_to_matrix(string)
        padding_length = padding_start - length

        # Fill the matrix with all 0s
        zeros_matrix = numpy.zeros([padding_length, string_matrix.shape[1]])
        # Concatenate the string matrix with zeros matrix. Padding the string matrix in the trailing
        string_matrix = numpy.concatenate([string_matrix, zeros_matrix], axis = 0)

        return string_matrix

def label_one_hot(labels):

    values = numpy.array(labels)
    maximum = numpy.max(values) + 1

    return numpy.eye(maximum)[values]

def setup():

    root = "../../../../Exclusion/Datasets/"

    stops, trains, tests = read(root = root)

    texts = []
    labels = []

    for title in trains["title"]:

        title = refine(title)
        title = padding_string_matrix(title)

        texts.append(title)

    for label in trains["label"]:

        label = numpy.int32(label)

        labels.append(label)

    texts = numpy.array(texts)

    labels = numpy.array(labels)
    labels = label_one_hot(labels)

    texts = numpy.expand_dims(texts, axis = -1)

    return texts, labels

def start():

    orginal = [1, 2, 3, 4, 5, 6, 0]

    eyes = one_hot(orginal)

    print(eyes)

    """
    
        [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0.]]
    
    Process finished with exit code 0

    """

    string = "hello"

    matrix = string_to_matrix(string)

    print("matrix =", matrix)

    trains, labels = setup()

    print("trains.shape = {}, labels.shape = {}".format(trains.shape, labels.shape))

def main():

    start()

if __name__ == "__main__":

    main()
