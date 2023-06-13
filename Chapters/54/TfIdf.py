import jax
import sys
import gensim

sys.path.append("../52/")
import AgNewsCsvReader

sys.path.append("../53/")
import Word2VecGenSim

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

def idf(corpus):
    
    dfi = {}
    N = 0.0
    
    # Number of occurences
    for document in corpus:
        
        N += 1
        
        counted = []
        
        for word in document:
            
            if word not in counted:
                
                counted.append(word)
                
                if word in dfi:
                    
                    dfi[word] += 1
                    
                else:
                    
                    dfi[word] = 1
    
    idfs = {}
    
    for word in dfi:
        
        idfs[word] = jax.numpy.log(N / float(dfi[word]))
        
    return idfs

def tfidf(corpus):
    
    tfs = {}
    tfidf_strings = []
    
    for document in corpus:
        
        word_occurences_in_given_document = {}
        
        for word in document:
            
            if word in tfs:
                word_occurences_in_given_document[word] += 1
            else:
                word_occurences_in_given_document[word] = 1
        
        idfs = idf(corpus)
        
        for word in tfs:
            
            word_occurences_in_given_document[word] *= idfs[word]
            sorted_values = sorted(word_occurences_in_given_document.items(), key = lambda  item: item[1], reverse = True)
            sorted_values = [value[0] for value in sorted_values]
            
            tfidf_strings.append(sorted_values)
            
    return tfidf_strings

def train():
    
    labels, titles, descriptions = AgNewsCsvReader.setup()
    
    name =  "/tmp/CorpusWord2Vec.bin"
    
    callback = LossLogger()
    
    descriptions = tfidf(descriptions)
    
    model = Word2VecGenSim.retrain(descriptions, name, callback)
   
    text = "Inspection now is in progress"
    
    Word2VecGenSim.vectorize(model, text)
    
def main():
    
    train()
    
if __name__ == "__main__":
    
    main()
