import csv
import re
import jax
import numpy
import jax.example_libraries.stax
import jax.example_libraries.optimizers

def setup():
    
    with open("../../Shares/ag_news_csv/train.csv", "r") as handler:
        
        train_labels = []
        train_texts = []
        
        trains = csv.reader(handler)
        trains = list(trains)
        
        for i in range(len(trains)):
            
            line = trains[i]
            
            train_labels.append(jax.numpy.int32(line[0]))
            train_texts.append(purify(line[1], split = True))
            
    return train_labels, train_texts

def purify(string: str, pattern: str = r"[^a-z]", replacement: str = " ", split = False):
    
    string = string.lower()
    
    string = re.sub(pattern = pattern, repl = replacement, string = string)
    # Replace the consucutive spaces with single space
    string = re.sub(pattern = r" +",  repl = replacement, string = string)
    # string = re.sub(pattern = " ", repl = "", string = string)
    
    # Trim the string
    string = string.strip()
    string = string + " eos"
    
    if split:
        
        string = string.split(" ")
    
    return string

def one_hot_numbers(numbers):
    
    array = numpy.array(numbers)
    maximum = numpy.max(array) + 1
    
    eyes = numpy.eye(maximum)[array]
    
    return eyes
    
def make_words_matrix(maximum_length = 12):
    
    train_labels, train_texts = setup()
    
    import gensim.models
        
    model = gensim.models.word2vec.Word2Vec(train_texts, vector_size = 64, min_count = 0, window = 5)
    
    matrixes = []
    
    for line in train_texts:
        
        length = len(line)
        
        if length > maximum_length:
            
            line = line[: maximum_length]
            matrix = model.wv[line]
            
            matrixes.append(matrix)
            
        else:
            
            matrix = model.wv[line]
            padding_length = maximum_length - length
            padding_matrix = numpy.zeros([padding_length, 64]) + 1e-10
            
            matrix = jax.numpy.concatenate([matrix, padding_matrix], axis = 0)
            
            matrixes.append(matrix)
            
    train_texts = numpy.expand_dims(matrixes, axis = 3)
    train_labels = one_hot_numbers(train_labels)
    
    return train_texts, train_labels

def cnn(number_classes):
    
    return jax.example_libraries.stax.serial(
        
        jax.example_libraries.stax.Conv(1, (3, 3)),
        jax.example_libraries.stax.Relu,
        
        jax.example_libraries.stax.Conv(1, (5, 5)),
        jax.example_libraries.stax.Relu,
        
        jax.example_libraries.stax.Flatten,
        
        jax.example_libraries.stax.Dense(32),
        jax.example_libraries.stax.Relu,
        
        jax.example_libraries.stax.Dense(number_classes),
        
        jax.example_libraries.stax.LogSoftmax
        
        )
    
def verify_accuracy(params, batch, predict_function):
    
    inputs, targets = batch
    predictions = predict_function(params, inputs)
    class_ = jax.numpy.argmax(predictions, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)
    
    return jax.numpy.sum(predictions == targets)

def loss_function(params, batch, predict_function):
    
    inputs, targets = batch
    
    predictions = predict_function(params, inputs)
    
    losses = -targets * predictions
    losses = jax.numpy.sum(losses, axis = 1)
    losses = jax.numpy.mean(losses)
    
    return losses

def update_function(i, optimizer_state, batch, get_params_function, optimizer_update_function, predict_function):
    
    params = get_params_function(optimizer_state)
    
    loss_function_grad = jax.grad(loss_function)
    gradients = loss_function_grad(params, batch, predict_function)
    
    return optimizer_update_function(i, gradients, optimizer_state)
    
def train():
    
    train_texts, train_labels = make_words_matrix()
    
    prng = jax.random.PRNGKey(15)
    
    number_classes = 5
    input_shape = [-1, 64, 28, 1]
    batch_size = 100
    epochs = 5
    
    init_random_params, predict = cnn(number_classes)
    
    optimizer_init_function, optimizer_update_function, get_params_function = jax.example_libraries.optimizers.adam(step_size = 2.17e-4)
    _, init_params = init_random_params(prng, input_shape = input_shape)
    optimizer_state = optimizer_init_function(init_params)
    
    print("train_texts.shape =", train_texts.shape, ", train_labels.shape =", train_labels.shape)
    
    train_batch_number = int(len(train_texts) / batch_size)
    
    for i in range(epochs):
        
        print(f"Epoch {i} started")
    
        for j in range(train_batch_number):
            
            start = j * batch_size
            end = (j + 1) * batch_size
            
            batch = (train_texts[start: end], train_labels[start: end])
            
            optimizer_state = update_function(i, optimizer_state, batch, get_params_function, optimizer_update_function, predict)
            
            if (j + 1) % 10 == 0:
                
                params = get_params_function(optimizer_state)
                losses = loss_function(params, batch)
                
                print("Losses now is =", losses)
                
        params = get_params_function(optimizer_state)
        
        print(f"Epoch {i} compeleted")
                 
if __name__ == "__main__":
    
    train()