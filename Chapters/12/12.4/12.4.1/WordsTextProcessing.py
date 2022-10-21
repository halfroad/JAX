import re
import sys

import numpy
from gensim.models import word2vec

sys.path.append("../../12.1/12.1.1/12-1/")

from DatasetsBuilder import read

def refine(text:str):

    text = text.lower()

    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r" +r", " ", text)

    text = text.strip()

    text = text + " eos"

    text = text.split(" ")

    return text

def setup():

    root = "../../../../Exclusion/Datasets/"

    stops, trains, tests = read(root = root)

    texts = []
    labels = []

    for title in trains["title"]:

        title = refine(title)

        texts.append(title)

    for label in trains["label"]:

        labels.append(numpy.int32(label))

    return texts, labels

def label_one_hot(labels):

    values = numpy.array(labels)
    maximum = numpy.max(values) + 1

    return numpy.eye(maximum)[values]


def get_words_vector(texts, labels, maximum_length = 12):

    # Set the parameters for model train
    model = word2vec.Word2Vec(texts, vector_size = 64, min_count = 1, window = 5)

    _texts = []

    for text in texts:

        length = len(text)

        if length > maximum_length:

            text = text[: maximum_length]
            vectors = model[text]

            _texts.append(vectors)

        else:

            vectors = model[text]
            padding_length = maximum_length - length

            padding_matrix = numpy.zeros([padding_length, 64]) + 1e-10
            vectors = numpy.concatenate([vectors, padding_matrix])

            texts.append(vectors)

    _texts = numpy.expand_dims(texts, 3)
    labels = label_one_hot(labels)

    return numpy.array(_texts), numpy.array(labels)

def start():

    texts, labels = setup()
    texts, labels = get_words_vector(texts, labels)

    print(texts.shape, labels.shape)

if __name__ == "__main__":

    start()
