import csv
import re
import ssl

import jax.numpy
import nltk
from nltk.corpus import stopwords


def setup():

    handle = open("../../../../../../../Shares/Datasets/agnews/train.csv", mode = "r")
    trains = csv.reader(handle)

    return trains

def text_purify(string: str):

    # Lower the string
    string = string.lower()

    # Replace the nonstandard alphabat, includes punctuation makes
    string = re.sub(r"[^a-z0-9]", repl = " ", string = string)

    # Replace multiple spaces
    string = re.sub(r" +", repl = " ", string = string)

    # Trim the spaces in head and tail
    string = string.strip()

    # Split the string with Space
    strings = string.split(" ")

    return strings

def split():

    labels = []
    titles = []
    descriptions = []

    trains = setup()

    for line in trains:

        label = re.findall(r'\d+\.\d+', line[0].strip())
        labels.append(jax.numpy.float32(label))

        titles.append(text_purify(line[1].lower()))
        descriptions.append(text_purify(line[2].lower()))

    print("labels = ", labels, "titles = ", titles, "descriptions = ", descriptions)

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

    root = "../../../../../../../Shares/Datasets/NLTK/"
    path = root + "stopwords"

    download_stop_words(path)

    stops = stopwords.words("english")

    print(stops)

def main():

    stop_words()

if __name__ == '__main__':

    main()
