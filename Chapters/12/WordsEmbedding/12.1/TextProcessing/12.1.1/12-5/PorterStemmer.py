import csv
import re
import ssl

import jax
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords

def setup():

    handle = open("../../../../../../../Shares/Datasets/agnews/train.csv", mode = "r")
    trains = csv.reader(handle)

    return trains

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

    return stops

def text_purify(string: str, stops):

    # Lower the string
    string = string.lower()

    # Replace the nonstandard alphabat, includes punctuation makes
    string = re.sub(r"[^a-z0-9]", repl = " ", string = string)

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

    return strings

def split(trains):

    labels = []
    titles = []
    descriptions = []

    stops = stop_words()

    for line in trains:

        label = re.findall(r'\d+\.\d+', line[0].strip())
        labels.append(jax.numpy.float32(label))

        titles.append(text_purify(line[1].lower(), stops = stops))
        descriptions.append(text_purify(line[2].lower(), stops = stops))

    print("labels = ", labels, "titles = ", titles, "descriptions = ", descriptions)

def main():

    trains = setup()
    split(trains)

if __name__ == '__main__':

    main()
