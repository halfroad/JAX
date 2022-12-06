import csv
import os.path
import re
import ssl

import jax
import nltk
from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
from nltk import PorterStemmer
from nltk.corpus import stopwords

def setup():

    handle = open("../../../../../../Shares/Datasets/agnews/train.csv", mode = "r")
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

    root = "../../../../../../Shares/Datasets//NLTK/"
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

    # Add the suffix of end
    strings = strings + ["eos"]

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

    return labels, titles, descriptions

class LossLogger(CallbackAny2Vec):

    """
    Output loss at each epoch
    """

    def __init__(self):

        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model_):

        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model_):

        loss = model_.get_latest_training_loss()

        self.losses.append(loss)

        print(f'  Loss: {loss}')

        self.epoch += 1

def train(name, sentences: [str], callback):

    # Set the params for training
    model = word2vec.Word2Vec(sentences, vector_size = 64, min_count = 0, window = 2, compute_loss = True, callbacks = [callback])

    model.save(fname_or_handle = name)

    return model

def retrain(name, sentences, callback):

    model = word2vec.Word2Vec.load(name)

    # Continue to train the model

    model.train(sentences, epochs = model.epochs, total_examples = model.corpus_count, callbacks = [callback])

    return model


def predict(model, strings):

    stops = stop_words()

    strings = text_purify(strings, stops = stops)

    print(strings)

    shape = model.wv[strings].shape

    print(model.wv[strings])

def main():

    trains = setup()

    labels, titles, descriptions = split(trains)

    # The name for model storage
    name = "../../../../../../Shares/Models/CorpusWord2Vec.bin"

    loss_logger = LossLogger()

    if os.path.exists(name):
        model = retrain(name, titles, callback = loss_logger)
    else:
        model = train(name, descriptions, callback = loss_logger)

    string = "Prediction Unit Helps Forecast Wildfires"

    predict(model, string)

if __name__ == '__main__':

    main()



