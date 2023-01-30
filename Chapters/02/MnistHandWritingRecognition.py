import tensorflow
import tensorflow_datasets
import jax

from jax.example_libraries import stax, optimizers

def setup():
    
    dataset, metadata = tensorflow_datasets.load(name = "mnist", split = [tensorflow_datasets.Split.TRAIN, tensorflow_datasets.Split.TEST], batch_size =-1, as_supervised = True, with_info = True, data_dir = "../../Shares/Datasets/MNIST/")
    
    (train_images, train_labels), (test_images, test_labels) = tensorflow_datasets.as_numpy(dataset)
    
    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)
    
    total_train_images = len(train_labels)
    
    trains = tensorflow.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1024).batch(256).prefetch(tensorflow.data.experimental.AUTOTUNE)
    tests = tensorflow.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(1024).batch(256).prefetch(tensorflow.data.experimental.AUTOTUNE)
    
    trains = tensorflow_datasets.as_numpy(trains)
    tests = tensorflow_datasets.as_numpy(tests)
    
    # Extract informative features
    class_names = metadata.features["label"].names
    number_classes = metadata.features["label"].num_classes
    
    reshape_arguments = [(-1, 28 * 28), (-1, )]
    input_shape = reshape_arguments[0]
    step_size = 1e-3
    epochs = 10
    batch_size = 128
    
    key = jax.random.PRNGKey(15)
    
    return (number_classes, class_names, input_shape, step_size, epochs, batch_size, key), (trains, tests, total_train_images)

def one_hot(inputs, size = 10, dtype = jax.numpy.float32):

    """
    
    One-hot encoding
    
    """
    
    return jax.numpy.array(inputs[:, None] == jax.numpy.arange(size), dtype = dtype)

def loss_function(params, batch, predict_function):

    """
    Cross Entropy Loss Function
    """
    
    inputs, targets = batch
    
    predictions = predict_function(params, inputs)
    
    entropys = jax.numpy.sum(-targets * predictions, axis = 1)
    entropys = jax.numpy.mean(entropys)
    
    return entropys
    
def update_function(i, optimizer_state, batch, get_params_function, update_function, predict_function):

    """
    Optimizer
    """
    
    params = get_params_function(optimizer_state)
    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, batch, predict_function)
    
    return update_function(i, gradients, optimizer_state)
    
def verify_accuracy(params, batch, predict_function):

    """
    Verify the predictions
    """
    
    inputs, targets = batch
    
    result = predict_function(params, inputs)
    classification = jax.numpy.argmax(result, axis = 1)
    target = jax.numpy.argmax(targets, axis = 1)
    
    return jax.numpy.sum(classification == target)
    
def model():

    """
    {Dense(1024) -> ReLU} x 2 -> Dense(10) -> LogSoftmax
    """
    
    init_random_params, predict_function = stax.serial(
        
        stax.Dense(1024),
        stax.Relu,
        
        stax.Dense(1024),
        stax.Relu,
        
        stax.Dense(10),
        stax.LogSoftmax
    )
    
    return init_random_params, predict_function

def train():

    (number_classes, class_names, input_shape, step_size, epochs, batch_size, key), (trains, tests, total_train_images) = setup()
    
    optimizer_init_function, optimizer_update_function, get_params_function = optimizers.adam(step_size = step_size)
    init_random_params, predict_function = model()
        
    _, init_params = init_random_params(key, input_shape)
    optimizer_state = optimizer_init_function(init_params)
    
    for i in range(epochs):
    
        iteration = 0
        
        for batch in trains:
        
            images = batch[0].reshape(input_shape)
            targets = batch[1].reshape((-1, 10))
            
            optimizer_state = update_function(iteration, optimizer_state, (images, targets), get_params_function, optimizer_update_function, predict_function)
            
            if (iteration + 1) % 100 == 0:
            
                print(f"Iteration: {iteration + 1} of epoch {i + 1} is completed")
            
            iteration += 1
            
        params = get_params_function(optimizer_state)
        
        accuracies = []
        predictions = 0.0
        
        for batch in trains:
        
            images = batch[0].reshape(input_shape)
            targets = batch[1].reshape((-1, 10))
            
            predictions += verify_accuracy(params, (images, targets), predict_function)
        
        accuracies.append(predictions / float(total_train_images))
        
        print(f"Training set accuracy: {accuracies} after {i + 1} epochs")
    
def main():
    
    train()
    
if __name__ == "__main__":
    
    main()
