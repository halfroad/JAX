import os
import sys

from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec

sys.path.append("../12.1.1/12-1/")

from AGNews import AGNewsDatasetAutoGenerator
from DatasetsBuilder import read

sys.path.append("../12.1.2/12-5/")
from TextClearence import text_clear


def build_model(texts, name):

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

    loss_logger = LossLogger()

    # Set the parameters for train
    model = word2vec.Word2Vec(texts, vector_size = 64, min_count = 0, window = 5, compute_loss = True, callbacks = [loss_logger])

    model.save(name)

    return model

def retrain(texts, name):

    model = word2vec.Word2Vec.load(name)
    model.train(texts, epochs = model.epochs, total_examples = model.corpus_count)

def start():

    stops, trains, tests = read(root = "../../../../Exclusion/Datasets/")
    name = "CorpusWord2Vector.bin"

    model = build_model(trains, name)

    text = "Prediction Unit Helps Forecast Wildfires"
    text = text_clear(text, stops)

    print(model.wv[text].shape)

def main():

    start()

if __name__ == "__main__":

    main()
