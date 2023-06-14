import tensorflow
import tensorflow_datasets
import jax
from jax.example_libraries import stax, optimizers

def one_hot(inputs, size = 10, dtype = jax.numpy.float32):

    """
    One-hot encoding
    """
    
    return jax.numpy.array(inputs[:, None] == jax.numpy.arange(size), dtype = dtype)
    
def setup():
    
    dataset, metadata = tensorflow_datasets.load(name = "mnist", split = [tensorflow_datasets.Split.TRAIN, tensorflow_datasets.Split.TEST], batch_size =-1, as_supervised = True, with_info = True, data_dir = "/tmp/")
    
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
    inputs_shape = reshape_arguments[0]
    step_size = 1e-3
    epochs = 10
    batch_size = 128
    
    prng = jax.random.PRNGKey(15)
    
    return (number_classes, class_names, step_size, epochs, inputs_shape, batch_size, prng), (trains, tests, total_train_images)

def loss_function(params, batch, predict_function):

    inputs, targets = batch
    predictions = predict_function(params, inputs)
    
    """
    Cross Entropy Loss Function
    """
    losses = jax.numpy.sum(-targets * predictions, axis = 1)
    losses = jax.numpy.mean(losses)
    
    return losses
    
def update_function(i, optimizer_state, batch, get_params_function, optimizer_update_function, predict_function):

    """
    Optimizer
    """
    
    params = get_params_function(optimizer_state)
    loss_function_grad = jax.grad(loss_function)
    
    gradients = loss_function_grad(params, batch, predict_function)
    
    return optimizer_update_function(i, gradients, optimizer_state)
    
def verify_accuracy(params, batch, predict_function):

    """
    Verify the predictions
    """
    
    inputs, targets = batch
    
    predictions = predict_function(params, inputs)
    classification = jax.numpy.argmax(predictions, axis = 1)
    target = jax.numpy.argmax(targets, axis = 1)
    
    return jax.numpy.sum(classification == target)
    
def model():

    """
    {Dense(1024) -> ReLU} x 2 -> Dense(10) - LogSoftmax
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

    (number_classes, class_names, step_size, epochs, inputs_shape, batch_size, prng), (trains, tests, total_train_images) = setup()
    
    optimizer_init_function, optimizer_update_function, get_params_function = optimizers.adam(step_size = step_size)
    init_random_params_function, predict_function = model()
    
    _, init_params = init_random_params_function(prng, inputs_shape)
    optimizer_state = optimizer_init_function(init_params)
    
    batch_number = (int)(total_train_images / batch_size)
    
    for i in range(epochs):
    
        iteration = 0
        
        for batch in trains:
        
            images = batch[0].reshape(inputs_shape)
            targets = batch[1].reshape((-1, 10))
            
            optimizer_state = update_function(iteration, optimizer_state, (images, targets), get_params_function, optimizer_update_function, predict_function)
            
            if (iteration + 1) % 100 == 0:
            
                print(f"Iteration: {iteration + 1} of epochs {i + 1} is completed.")
                
            iteration += 1
            
        params = get_params_function(optimizer_state)
        
        accuracies = []
        predictions = 0.0
        
        for batch in trains:
        
            images = batch[0].reshape(inputs_shape)
            targets = batch[1].reshape((-1, 10))
            
            predictions += verify_accuracy(params, (images, targets), predict_function)
            
        accuracies.append(predictions / float(total_train_images))
        
        print(f"Training set accuracies: {accuracies} after {i + 1} epochs.")
            
def main():
    
    train()
    
if __name__ == "__main__":

    main()
    
