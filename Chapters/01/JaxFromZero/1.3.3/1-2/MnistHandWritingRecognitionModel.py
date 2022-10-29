import jax.numpy
import tensorflow
import tensorflow_datasets
from jax.example_libraries import stax, optimizers


def setup():

    dataset, metadata = tensorflow_datasets.load(name = "mnist", data_dir = "../../../../Shares/Datasets/MNIST/", split = [tensorflow_datasets.Split.TRAIN, tensorflow_datasets.Split.TEST], batch_size = -1, as_supervised = True, with_info = True)
    (train_images, train_labels), (test_images, test_labels) = tensorflow_datasets.as_numpy(dataset)

    train_labels = one_hot_no_jit(train_labels)
    test_labels = one_hot_no_jit(test_labels)

    total_train_images = len(train_labels)

    trains = tensorflow.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1024).batch(256).prefetch(tensorflow.data.experimental.AUTOTUNE)
    tests = tensorflow.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(1024).batch(256).prefetch(tensorflow.data.experimental.AUTOTUNE)

    trains = tensorflow_datasets.as_numpy(trains)
    tests = tensorflow_datasets.as_numpy(tests)

    # Extract informative features
    class_names = metadata.features["label"].names
    number_classes = metadata.features["label"].num_classes

    reshape_arguments = [(-1, 28 * 28), (-1,)]
    input_shape = reshape_arguments[0]
    step_size = 1e-3
    epochs = 10
    batch_size = 128
    momentum_mass = 0.9

    key = jax.random.PRNGKey(0)

    return (number_classes, class_names, input_shape, step_size, epochs, batch_size, momentum_mass, key), (trains, tests, total_train_images)

def one_hot_no_jit(inputs, k = 10, dtype = jax.numpy.float32):

    """

    Create a one-hot encoding of inputs of size k.

    """

    return jax.numpy.array(inputs[:, None] == jax.numpy.arange(k), dtype = dtype)

def loss(parameters, batch, predict_function):

    """

    Cross-entropy loss over a mini batch

    """

    inputs, targets = batch

    prediction = predict_function(parameters, inputs)

    entropys = jax.numpy.sum(-targets * prediction, axis = 1)
    entropys = jax.numpy.mean(entropys)

    return entropys

def update(i, state, batch, get_parameters_function, update_function, predict_function):

    """

    Single optimization step over a mini batch.

    """

    parameters = get_parameters_function(state)
    grad_loss = jax.grad(loss)
    gradient = grad_loss(parameters, batch, predict_function)

    return update_function(i, gradient, state)

def verify_accuracy(parameters, batch, predict_function):

    """

    Verify the predictions over a mini batch

    """

    inputs, targets = batch

    result = predict_function(parameters, inputs)
    class_ = jax.numpy.argmax(result, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)

    return jax.numpy.sum(class_ == targets)

def model():

    """

    {Dense(1024) -> ReLU} X 2 -> Dense(10) -> Lohsoftmax

    """

    init_random_parameters, predict = stax.serial(

        stax.Dense(1024),
        stax.Relu,

        stax.Dense(1024),
        stax.Relu,

        stax.Dense(10),
        stax.LogSoftmax
    )

    return init_random_parameters, predict

def train():

    (number_classes, class_names, input_shape, step_size, epochs, batch_size, momentum_mass, key), (trains, tests, total_train_images) = setup()

    """
    
    train_images = train_images[: 1000]
    train_labels = train_labels[: 1000]
    
    """



    # step_size is learning_rate
    init_function, update_function, get_parameters_function = optimizers.adam(step_size = step_size)
    init_random_parameters, predict = model()

    _, init_parameters = init_random_parameters(key, input_shape)
    state = init_function(init_parameters)

    for i in range(epochs):

        iteration = 0

        for batch in trains:

            entry = batch[0].reshape(input_shape)
            targets = batch[1].reshape((-1, 10))

            state = update(iteration, state, (entry, targets), get_parameters_function, update_function, predict)

            if (iteration + 1) % 100 == 0:
                print(f"Iteration: {iteration + 1} of epoch {i + 1}")

            iteration += 1


        parameters = get_parameters_function(state)

        accuracies = []
        verified_predictions = 0.0

        for batch in trains:

            entry = batch[0].reshape(input_shape)
            targets = batch[1].reshape((-1, 10))

            verified_predictions += verify_accuracy(parameters, (entry, targets), predict)

        accuracies.append(verified_predictions / float(total_train_images))

        print(f"Training set accuracy: {accuracies} after {i + 1} epochs")


def start():

    train()

if __name__ == "__main__":

    start()
