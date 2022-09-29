import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from collections import namedtuple

from jax import jit, grad, random
from jax.example_libraries import optimizers, stax

def get_configurations():
    
    Configuration = namedtuple("Configuration", ["classifications_number", "reshape_arguments", "input_shape", "learning_rate", "epochs", "batch_size", "momentum_mass", "prng"])
    reshape_arguments = [(-1, 28 * 28), (-1,)]
    configuration = Configuration(classifications_number = 10,
                                  reshape_arguments = reshape_arguments,
                                  input_shape = reshape_arguments[0],
                                  learning_rate = 0.001,
                                  epochs = 10,
                                  batch_size = 128,
                                  momentum_mass = 0.9,
                                  prng = random.PRNGKey(0))
    
    return configuration

def prepare(batch_size):

    tfds.core.utils.gcs_utils._is_gcs_disabled = True
    # tfds.display_progress_bar(enable = True)
    
    print(tfds.list_builders())
    
    dataset = tfds.load(name ="mnist", split = [tfds.Split.TRAIN, tfds.Split.TEST], data_dir = "../Exclusion/Datasets/MNIST/", batch_size = -1, as_supervised = True)
    (images_train, labels_train), (images_test, labels_test) = tfds.as_numpy(dataset)
    
    print("images_train.shape = {}, labels_train.shape = {}, images_test.shape = {}, labels_test.shape = {}".format(images_train.shape, labels_train.shape, images_test.shape, labels_test.shape))
    
    total_train_images = len(labels_train)
    
    # images_train = one_hot_nojit(images_train)
    labels_train = one_hot_nojit(labels_train)
    
    dataset_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train)).shuffle(1024).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    dataset_train = tfds.as_numpy(dataset_train)
    
    print("dataset_train = {}, total_train_images = {}".format(len(dataset_train), total_train_images))
    
    return dataset_train, total_train_images

def one_hot_nojit(x, k = 10, dtype = jnp.float32):

    """

    Create a one-hot encoding of x of k.

    """

    return jnp.array(x[:, None] == jnp.arange(k), dtype)

# {Dense(1024) -> ReLU} x 2 -> Dense(10) -> Logsoftmax

def check_prediction(parameters, batch, predict_function):
    
    """

    Correct predictions over a mini batch
    
    """
    
    inputs, targets = batch
    prediction = predict_function(parameters, inputs)
    classification = jnp.argmax(prediction, axis = 1)
    targets = jnp.argmax(targets, axis = 1)
    
    return jnp.sum(classification == targets)

# {Dense(1024) -> ReLU} X 2 -> Dense(10) -> Logsoftmax
    
def create_model():
    
    initial_random_parameters_function, predict_function = stax.serial(
        
        stax.Dense(1024),
        stax.Relu,
        stax.Dense(1024),
        stax.Relu,

        stax.Dense(10),
        stax.LogSoftmax)
    
    return initial_random_parameters_function, predict_function

def create_optimizer(initial_random_parameters_function, input_shape, prng):
    
    optimizer_initialize, optimizer_update_function, get_paremeters_function = optimizers.adam(step_size = 2e-4)
    _, initial_parameters = initial_random_parameters_function(prng, input_shape)
    optimizer_state = optimizer_initialize(initial_parameters)
    
    return optimizer_update_function, get_paremeters_function, optimizer_state

def compute_loss(parameters, batch, predict_function):
    
    """

    Cross-Entropy loss over a mini batch.
    
    """
    
    inputs, targets = batch
    
    return jnp.mean(jnp.sum(-targets * predict_function(parameters, inputs), axis = 1))

def update(i, optimizer_state, batch, optimizer_update_function, get_paremeters_function, predict_function):
    
    """

    Single optimization over a mini batch.

    """
    
    parameters = get_paremeters_function(optimizer_state)
    
    return optimizer_update_function(i, grad(compute_loss)(parameters, batch, predict_function), optimizer_state)
    
def train_model(dataset_train, optimizer_state, total_train_images, get_paremeters_function, optimizer_update_function, predict_function):
    
    counter = 0
    
    for i in range(17):
        
        for batch in dataset_train:
            
            data = batch[0].reshape((-1, 28 * 28))
            targets = batch[1].reshape((-1, 10))
            optimizer_state = update((counter), optimizer_state, (data, targets), optimizer_update_function, get_paremeters_function, predict_function)
            
            counter += 1
            
        parameters = get_paremeters_function(optimizer_state)
        
        accuracies = []
        correct_prediction = 0.0
        
        for batch in dataset_train:
            
            data = batch[0].reshape((-1, 28 * 28))
            targets = batch[1].reshape((-1, 10))
            
            correct_prediction += check_prediction(parameters, (data, targets), predict_function)
            
        accuracies.append(correct_prediction / float(total_train_images))
        
        print(f"Training set accuracy: {accuracies}")
    
    
def start():
    
    configuration = get_configurations()
    dataset_train, total_train_images = prepare(configuration.batch_size)
    initial_random_parameters_function, predict_function = create_model()
    optimizer_update_function, get_paremeters_function, optimizer_state = create_optimizer(initial_random_parameters_function, configuration.input_shape, configuration.prng)
    
    train_model(dataset_train, optimizer_state, total_train_images, get_paremeters_function, optimizer_update_function, predict_function)
    
    
def main():

    start()

if __name__ == "__main__":

    main()
