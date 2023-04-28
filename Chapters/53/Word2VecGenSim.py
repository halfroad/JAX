import sys
import gensim

sys.path.append("../52/")

import AgNewsCsvReader

class LossLogger(gensim.models.callbacks.CallbackAny2Vec):
    
    """"

    Output loss at each epoch

    """
    
    def __init__(self):
        
        self.epoch = 1
        self.losses = []
    
    def on_train_begin(self,model):
        
        print("Train started")
        
    def on_epoch_begin(self, model):
        
        print(f"Epoch {self.epoch}", end = '\t')
        
    def on_epoch_end(self, model):
        
        loss = model.get_latest_training_loss()
        
        self.losses.append(loss)
        
        print(f"Loss: {loss}")
        
        self.epoch += 1
        
    def on_train_end(self, model):
        
        print("Train ended")

def train(strings, name, callback = LossLogger()):
    
    model = gensim.models.word2vec.Word2Vec(strings, vector_size = 64, min_count = 0, window = 5, callbacks = [callback])
    name = "/tmp/CorpusWord2Vec.bin"
    
    model.save(name)
    
    return model
    
def retrain(strings, name, callback = LossLogger()):
    
    model = gensim.models.word2vec.Word2Vec.load(name)
    
    model.train(strings, epochs = model.epochs, total_examples = model.corpus_count, callbacks = [callback])
    
    return model
    
def vectorize(model, string):
    
    string = AgNewsCsvReader.purify(string)
    
    print(model.wv[string])
    
def main():
    
    labels, titles, descriptions = AgNewsCsvReader.setup()
    
    name =  "/tmp/CorpusWord2Vec.bin"
    
    callback = LossLogger()
    
   # train(titles, name, callback)
    model = retrain(descriptions, name, callback)
    
    text = "Inspection now is in progress"
    
    vectorize(model, text)
    
if __name__ == "__main__":
    
    main()
