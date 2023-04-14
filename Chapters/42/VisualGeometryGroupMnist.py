import jax
import tensorflow_datasets as tfds
import time

import MnistDatasets

def setup():

   train_images, train_labels, test_images, test_labels = MnistDatasets.mnist()
   
   batch_size = 600
   
   inputs_channels = 1
   epochs = 9
   
   prng = jax.random.PRNGKey(15)
   
   kernel_shapes = [
        [3, 3, 1, 16],
        [3, 3, 16, 32],
        [3, 3, 32, 48],
        [3, 3, 48, 64],
        [50176, 128],
        [128, 10]
    ]
   
   return (train_images, train_labels), (test_images, test_labels), (batch_size, epochs, prng, kernel_shapes)

def one_hot(inputs, length = 10, dtype = jax.numpy.float32):
    
    matches = jax.numpy.array(inputs[:, None] == jax.numpy.arange(length), dtype)
    
    return matches

def batch_normalize(inputs, beta = 0.25, epsilon = 1e-9, gamma = 0.9):
    
    u = jax.numpy.mean(inputs, axis = 0)
    std = jax.numpy.sqrt(inputs.var(axis = 0) + epsilon)
    y = (inputs - u) / std
    
    y_hat = gamma * y + beta
    
    return y_hat

def partial_flatten(inputs):

    """

    Flatten all but the first dimension of an array

    jax.lax.expand_dims(inputs, [-1]): [60000, 28, 28] -> [60000, 28, 28, 1]
    jax.lax.expand_dims(inputs, [1, 2]): [60000, 28, 28] -> [60000, 1, 1, 28, 28]

    """
    inputs = jax.lax.expand_dims(inputs, [-1])  # [60000, 28, 28] -> [60000, 28, 28, 1]

    return inputs / 255.


def init_mlp_params(shapes, prng):
    
    params = []
    
    # Create 12 layers kernels for Convolutional Neural Networks
    for i in range(len(shapes) - 2):
        
        weights = jax.random.normal(key = prng, shape = shapes[i]) / jax.numpy.sqrt(28. * 28.)
        
        _dict = dict(weight = weights)
        
        params.append(_dict)
         
    # Create 3 layers kernels for Dense
    for i in range(len(shapes) - 2, len(shapes)):
        
        weights = jax.random.normal(key = prng, shape = shapes[i]) / jax.numpy.sqrt(28. * 28.)
        biases = jax.random.normal(key = prng, shape = (shapes[i][-1],)) / jax.numpy.sqrt(28. * 28.)
        
        _dict = dict(weight = weights, bias = biases)
        
        params.append(_dict)
        
    return params

def conv(inputs, kernel, window_strides = 1):
    
    shape = inputs.shape
    dimension_numbers = jax.lax.conv_dimension_numbers(lhs_shape = shape, rhs_shape = kernel["weight"].shape, dimension_numbers = ("NHWC", "HWIO", "NHWC"))
    
    inputs = jax.lax.conv_general_dilated(inputs, kernel["weight"], window_strides = [window_strides, window_strides], padding = "SAME", dimension_numbers = dimension_numbers)
    inputs = jax.nn.selu(inputs)
    
    return inputs

@jax.jit
def forward(parameters, inputs):

    for i in range(len(parameters) - 2):

        inputs = conv(inputs, kernel = parameters[i])

    # inputs = BatchNormalization.batch_normalize(inputs)
    inputs = jax.numpy.reshape(inputs, newshape = (inputs.shape[0], 50176))

    for i in range(len(parameters) - 2, len(parameters) - 1):

        inputs = jax.numpy.matmul(inputs, parameters[i]["weight"]) + parameters[i]["bias"]
        inputs = jax.nn.selu(inputs)

    inputs = jax.numpy.matmul(inputs, parameters[-1]["weight"]) + parameters[-1]["bias"]
    inputs = jax.nn.softmax(inputs, axis = -1)

    return inputs

@jax.jit
def cross_entropy(genuines, predictions):
    
    entropys = genuines * jax.numpy.log(jax.numpy.clip(predictions, 1e-9, 0.999)) + (1 - genuines) * jax.numpy.log(jax.numpy.clip(1 - predictions, 1e-9, 0.999))
    entropys = jax.numpy.sum(entropys, axis = 1)
    entropys = jax.numpy.mean(entropys)
    
    return entropys

@jax.jit
def loss_function(parameters, inputs, genuines):

    predictions = forward(parameters, inputs)
    entropys = cross_entropy(genuines, predictions)

    return entropys

@jax.jit
def optimizer_function(parameters, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(parameters, inputs, genuines)

    new_parameters = jax.tree_util.tree_map(lambda parameter, gradient: parameter - learning_rate * gradient, parameters, gradients)

    return new_parameters

@jax.jit
def verify_accuracy(params, inputs, targets):
    
    """
    Correct predictions over a mini batch
    """
    predictions = forward(params, inputs)
    _class = jax.numpy.argmax(predictions, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)
    
    return jax.numpy.sum(_class == targets)

def train():
    
    (train_images, train_labels), (test_images, test_labels), (batch_size, epochs, prng, kernel_shapes) = setup()
    
    print(f"train_images.shape = {train_images.shape}, train_labels.shape = {train_labels.shape}), (test_images.shape = {test_images.shape}, test_labels.shape = {test_labels.shape}")
    
    '''
    train_images.shape = (60000, 28, 28), train_labels.shape = (60000,)), (test_images.shape = (10000, 28, 28), test_labels.shape = (10000,)
    '''
    
    train_images = partial_flatten(train_images)
    train_labels = one_hot(train_labels)
    
    test_images = partial_flatten(test_images)
    test_labels = one_hot(test_labels)
    
    params = init_mlp_params(kernel_shapes, prng)
    
    begin = time.time();
    
    for i in range(epochs):
        
        batch_number = train_images.shape[0] // batch_size
                
        for j in range(batch_number):
            
            start = batch_size * j
            stop = batch_size * (j + 1)
            
            images_batch = train_images[start: stop]
            labels_batch = train_labels[start: stop]
            
            params = optimizer_function(params, images_batch, labels_batch)
            
            print(f"Bacth number {j + 1}/{batch_number} within epoch {i + 1}/{epochs} is completed")
            
        if (i + 1) % 5 == 0:
            
            loss = loss_function(params, train_images, train_labels)
            
            end = time.time()
            
            accuracies = verify_accuracy(params, test_images, test_labels) / float(4096.)
            
            print(f"Now the loss is {loss}, accuracy is {accuracies} after {1 + 1} iterations")
            
            start = time.time()
            
if __name__ == "__main__":
    
    train()