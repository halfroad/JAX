import os
import sys

from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec

sys.path.append("../12.1.1/12-1/")

from AGNews import AGNewsDatasetAutoGenerator
from DatasetsBuilder import read

sys.path.append("../12.1.2/12-5/")
from TextClearence import text_clear

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

def build_model(texts, name, callback):

    # Set the parameters for train
    model = word2vec.Word2Vec(texts, vector_size = 64, min_count = 0, window = 5, compute_loss = True, callbacks = [callback])

    model.save(fname_or_handle = name)

    return model

def load(name):

    return word2vec.Word2Vec.load(name)

def retrain(texts, name, callback):

    model = load(name)
    model.train(texts, epochs = model.epochs, total_examples = model.corpus_count, callbacks = [callback])

    return model
def start():

    root = "../../../../Exclusion/Datasets/"

    stops, trains, tests = read(root = root)

    name = "../../../../Exclusion/Models/Word2Vector/CorpusWord2Vector.bin"

    loss_logger = LossLogger()

    texts = trains["description"]

    if os.path.exists(name):
        model = load(name)
    else:
        model = build_model(texts, name, callback = loss_logger)

    string = "Prediction Unit Helps Forecast Wildfires"
    string = text_clear(string, stops)

    print(model.wv[string])

    texts = trains["title"]

    model = retrain(texts, name, loss_logger)

    string = "Inspection now is in progress"
    string = text_clear(string, stops)

    print(model.wv[string])

def main():

    start()

if __name__ == "__main__":

    main()
