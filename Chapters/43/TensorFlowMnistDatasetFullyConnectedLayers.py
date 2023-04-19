import jax
import tensorflow_datasets as tfds
import time

def one_hot(inputs, depth, dtype = jax.numpy.float32):
    
    matches = jax.numpy.array(inputs[:, None] == jax.numpy.arange(depth), dtype = dtype)
    
    return matches

def setup():
    
    trains, tests = tfds.load("mnist", data_dir = "/tmp/", split = [tfds.Split.TRAIN, tfds.Split.TEST], batch_size = -1)
    
    trains = tfds.as_numpy(trains)
    tests = tfds.as_numpy(tests)
    
    train_images, train_labels = trains["image"], trains["label"]
    test_images, test_labels = tests["image"], tests["label"]
    
    _, height, width, channels = train_images.shape
    number_pixels = height * width * channels
    
    train_images = jax.numpy.reshape(train_images, (-1, number_pixels))
    test_images = jax.numpy.reshape(test_images, (-1, number_pixels))
    
    output_dimensions = 10
    
    train_labels = one_hot(train_labels, depth = output_dimensions)
    test_labels = one_hot(test_labels, depth = output_dimensions)
    
    key = jax.random.PRNGKey(15)
    epochs = 500
    
    return (number_pixels, output_dimensions, key, epochs), (train_images, train_labels), (test_images, test_labels)

def init_mpl_params(key, number_pixels, output_dimensions):
    
    def init(layer_dimensions = [number_pixels, 512, 256, output_dimensions]):

        parameters = []

        for i in range(1, (len(layer_dimensions))):

            weights = jax.random.normal(key, shape = (layer_dimensions[i - 1], layer_dimensions[i])) / jax.numpy.sqrt(number_pixels)
            biases = jax.random.normal(key, shape = (layer_dimensions[i],)) / jax.numpy.sqrt(number_pixels)

            _dict = {"weight": weights, "bias": biases}

            parameters.append(_dict)

        return parameters

    return init()

@jax.jit
def forward(params, inputs):
    
    for param in params[: -1]:
        
        weights = param["weight"]
        biases = param["bias"]
        
        inputs = jax.numpy.dot(inputs, weights) + biases
        inputs = relu(inputs)
        
    outputs = jax.numpy.dot(inputs, params[-1]["weight"]) + params[-1]["bias"]
    outputs = jax.nn.softmax(outputs, axis = -1)
    
    return outputs

@jax.jit
def relu(inputs):
    
    return jax.numpy.maximum(0, inputs)


@jax.jit
def cross_entropy(genuines, predictions):
    
    entropys = genuines * jax.numpy.log(jax.numpy.clip(predictions, 1e-9, 0.999)) + (1 - genuines) * jax.numpy.log(jax.numpy.clip(1 - predictions, 1e-9, 0.999))
    entropys = -jax.numpy.sum(entropys, axis = 1)
    
    return jax.numpy.mean(entropys)

@jax.jit
def loss_function(params, inputs, genuines):
    
    predictions = forward(params, inputs)
    
    return cross_entropy(genuines, predictions)

@jax.jit
def optimizer_function(params, inputs, genuines, learn_rate = 1e-3):
    
    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)
    
    params = jax.tree_util.tree_map(lambda param, gradient: param - learn_rate * gradient, params, gradients)
    
    return params

@jax.jit
def verify_accuracy(params, inputs, targets):
    
    predictions = forward(params, inputs)
    _classes = jax.numpy.argmax(predictions, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)
    
    return jax.numpy.sum(_classes == targets)
        
def train():
    
    (number_pixels, output_dimensions, key, epochs), (train_images, train_labels), (test_images, test_labels) = setup()
    
    print((train_images.shape, train_labels.shape), (test_images.shape, test_labels.shape))
    
    params = init_mpl_params(key = key, number_pixels = number_pixels, output_dimensions = output_dimensions)
    start = time.time()
    
    for i in range(epochs):
        
        params = optimizer_function(params, train_images, train_labels)
        
        if (i + 1) % 100 == 0:
            
            losses = loss_function(params, test_images, test_labels)
            end = time.time()
            
            accuracies = verify_accuracy(params, test_images, test_labels) / float(10000.)
            
            print("Time consumed: %.12f seconds" % (end - start), f"after {i + 1} iterations, now the loss is {losses}, accuracy of test set is {accuracies}")
            
            start = time.time()
    
if __name__ == "__main__":
    
    train()