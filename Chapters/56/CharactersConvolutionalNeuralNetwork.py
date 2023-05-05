import numpy
import jax
import jax.example_libraries.stax
import jax.example_libraries.optimizers
import sys

sys.path.append("../52/")

import AgNewsCsvReader


def one_hot(characters, alphabet):
        
    array = numpy.array(characters)
    length = len(alphabet)
    # jax.numpy.eye(N, M = None, K = 0, dtype) to create a 2-dimension array that
    # the elements in diagonal will be filled out with 1s, others are 0s.
    eyes = numpy.eye(length)[array]
    
    return eyes

def one_hot_numbers(numbers):
    
    array = numpy.array(numbers)
    maximum = numpy.max(array) + 1
    
    eyes = numpy.eye(maximum)[array]
    
    return eyes

def indexes_of(characters, alphabet):
        
    indexes = []
    
    for character in characters:
        
        index = alphabet.index(character)
        
        indexes.append(index)
        
    return indexes

def indexes_matrix(string, alphabet):
    
    indexes = indexes_of(string, alphabet)
    matrix = one_hot(indexes, alphabet)
    
    return matrix

def align_string_matrix(string, maximum_length = 64, alphabet = "abcdefghijklmnopqrstuvwxyz "):
    
    length = len(string)
    
    if length > maximum_length:
        
        string = string[: maximum_length]
        matrix = indexes_matrix(string, alphabet)
        
        return matrix
    
    else:
        
        matrix = indexes_matrix(string, alphabet)
        length = maximum_length - length
        matrix_padded = numpy.zeros([length, len(alphabet)])
        
        matrix = numpy.concatenate([matrix, matrix_padded], axis = 0)
        
        return matrix
    
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

def setup():
    
    prng = jax.random.PRNGKey(15)
    
    (train_labels, train_titles, train_descriptions), (test_labels, test_titles, test_descriptions) = AgNewsCsvReader.setup()
    
    train_texts = []
    
    for title in train_titles:
    
        matrix = align_string_matrix(title)
        
        train_texts.append(matrix)
        
    train_texts = numpy.expand_dims(train_texts, axis = -1)
    train_labels = one_hot_numbers(train_labels)
    
    test_texts = []
    
    for title in test_titles:
    
        matrix = align_string_matrix(title)
        
        test_texts.append(matrix)
        
    test_texts = numpy.expand_dims(test_texts, axis = -1)
    test_labels = one_hot_numbers(test_labels)
    
    number_classes = 5
    input_shape = [-1, 64, 28, 1]
    batch_size = 100
    epochs = 5
    
    init_random_params, predict = cnn(number_classes)
    
    optimizer_init_function, optimizer_update_function, get_params_function = jax.example_libraries.optimizers.adam(step_size = 2.17e-4)
    _, init_params = init_random_params(prng, input_shape = input_shape)
    optimizer_state = optimizer_init_function(init_params)
    
    return (prng, number_classes, batch_size, epochs, init_params, optimizer_state), (init_random_params, optimizer_init_function, predict, optimizer_update_function, get_params_function), ((train_texts, train_labels), (test_texts, test_labels)) 
    
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
    
    (prng, number_classes, batch_size, epochs, init_params, optimizer_state), (init_random_params, optimizer_init_function, predict, optimizer_update_function, get_params_function), ((train_texts, train_labels), (test_texts, test_labels)) = setup()
    
    print("train_texts.shape =", train_texts.shape, ", train_labels.shape =", train_labels.shape, ", test_texts.shape =", test_texts.shape, ", test_labels.shape =", test_labels.shape)
    
    train_batch_number = len(train_texts) / batch_size
    test_batch_number = len(test_texts) / batch_size
    
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
        
        accuracies = []
        predictions = 0.0
        
        for j in range(test_batch_number):
            
            start = j * batch_size
            end = (j + 1) * batch_size
            
            batch = (test_texts[start: end], test_labels[start: end])
            
            predictions += verify_accuracy(params, batch)
            
        accuracies.append(predictions / float(len(train_texts)))
        
        print(f"Training accuracies =", accuracies)
                 
if __name__ == "__main__":
    
    train()