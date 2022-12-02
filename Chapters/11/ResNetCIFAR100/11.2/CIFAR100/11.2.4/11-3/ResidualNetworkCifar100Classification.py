import jax.numpy
import numpy
import os.path
import pickle

from jax.example_libraries import stax, optimizers


def ConvBlock(kernel_size, filters, strides = (1, 1)):

    kernel_size_ = kernel_size

    # Set the numbers of convolutional kernel.
    filters1, filters2, filters3 = filters

    # Generate the main path
    Main = stax.serial(

        stax.Conv(filters1, (1, 1), strides, padding = "SAME"),
        stax.BatchNorm(),
        stax.Relu,

        stax.Conv(filters2, (kernel_size_, kernel_size_), padding = "SAME"),
        stax.BatchNorm(),
        stax.Relu,

        stax.Conv(filters3, (1, 1), padding = "SAME"),
        stax.BatchNorm()
    )

    Shortcut = stax.serial(

        stax.Conv(filters3, (1, 1), strides, padding = "SAME"),
        stax.BatchNorm()
    )

    return stax.serial(

        stax.FanOut(2),
        stax.parallel(Main, Shortcut),
        stax.FanInSum,
        stax.Relu
    )

def IdentityBlock(kernel_size, filters):

    kernel_size_ = kernel_size
    filters1, filters2 = filters

    # Generate the main path at first, here the dynamic self-assigned dimensions is used to adjust the dimensions
    def make_main(input_shape):

        return stax.serial(

            stax.Conv(filters1, (1, 1), padding = "SAME"),
            stax.BatchNorm(),
            stax.Relu,

            stax.Conv(filters2, (kernel_size_, kernel_size_), padding = "SAME"),
            stax.BatchNorm(),
            stax.Relu,

            # Adjust the dimensions dynamically relies on the input shape
            stax.Conv(input_shape[3], (1, 1), padding = "SAME"),
            stax.BatchNorm()
        )

    # Explicitly pass the size of dynamic input dimensions required by the model
    Main = stax.shape_dependent(make_main)

    # Combine the different computation channels
    return stax.serial(stax.FanOut(2),
                       stax.parallel(Main, stax.Identity),
                       stax.FanInSum,
                       stax.Relu)


def ResNet50(number_classes: int):

    return stax.serial(

        stax.Conv(64, (3, 3), padding = "SAME"),
        stax.BatchNorm(),
        stax.Relu,

        stax.MaxPool((3, 3), strides = (2, 2)),

        ConvBlock(3, [64, 64, 256]),

        IdentityBlock(3, [64, 64]),
        IdentityBlock(3, [64, 64]),

        ConvBlock(3, [128, 128, 512]),

        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),

        ConvBlock(3, [256, 256, 1024]),

        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),

        ConvBlock(3, [512, 512, 2048]),

        IdentityBlock(3, [512, 512]),
        IdentityBlock(3, [512, 512]),

        stax.AvgPool((7, 7)),

        stax.Flatten,
        stax.Dense(number_classes),
        stax.LogSoftmax
    )

# Function computes the accuracy
def verify_accuracy(params, batch, predict):

    """
    Correct predictions over a mini batch.
    """
    # Here 2 jax.numpy.argmax used for the convertion
    inputs, targets = batch

    predictions = predict(params, inputs)
    class_ = jax.numpy.argmax(predictions, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)

    accumlations = jax.numpy.sum(class_ == targets)

    return accumlations

# Function compute the loss
def loss_function(params, batch, predict):

    inputs, targets = batch
    losses = -targets * predict(params, inputs)

    losses = jax.numpy.sum(losses)
    losses = jax.numpy.mean(losses)

    return losses

# Function updates the params
def update(i, optimizer_state, batch, predict, get_params, optimizer_update):

    """
    Single optimization over a mini batch.
    """
    params = get_params(optimizer_state)

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, batch, predict)

    params = optimizer_update(i, gradients, optimizer_state)

    return params

def setup(root = "../../../../../../../Shares/Datasets/cifar-10-batches-py/"):

    def load(file_name: str):

        name = os.path.join(root, file_name)

        with open(name, "rb") as handle:

            data = pickle.load(handle, encoding = "latin1")

            return data

    batch1 = load("data_batch_1")
    batch2 = load("data_batch_2")
    batch3 = load("data_batch_3")
    batch4 = load("data_batch_4")
    batch5 = load("data_batch_5")

    train_images = []
    train_labels = []

    for batch in [batch1, batch2, batch3, batch4, batch5]:

        image = (batch["data"]) / 255.
        label = (batch["labels"])

        train_images.append(image)
        train_labels.append(label)

    train_images = numpy.concatenate(train_images)
    train_labels = numpy.concatenate(train_labels)

    batch = load("test_batch")

    test_images = []
    test_labels = []

    for data_ in [batch]:

        image = (data_["data"])
        label = (data_["labels"])

        test_images.append(image)
        test_labels.append(label)

    test_images = numpy.concatenate(test_images)
    test_labels = numpy.concatenate(test_labels)

    # Reshape the dimensiion from [n, channels, height, width] to [n, height, width, channels]
    train_images = jax.numpy.reshape(train_images, [-1, 3, 32, 32])
    train_images = jax.numpy.transpose(train_images, [0, 2, 3, 1])

    train_labels = jax.nn.one_hot(train_labels, num_classes = 100)

    test_images = jax.numpy.reshape(test_images, [-1, 3, 32, 32])
    test_images = jax.numpy.transpose(test_images, [0, 2, 3, 1])

    test_labels = jax.nn.one_hot(test_labels, num_classes = 100)

    init_random_params, predict = ResNet50(100)

    key = jax.random.PRNGKey(15)
    input_shape = [-1, 32, 32, 3]

    # step_size is the Learning Rate
    step_size = 2e-4

    optimizer_init, optimizer_update, get_params = optimizers.adam(step_size = step_size)
    _, init_params = init_random_params(key, input_shape)

    optimizer_state = optimizer_init(init_params)

    # Adjust relies on the formance of your computer
    batch_size = 128
    # Can be entire dataset if the performance is high
    total_number = 12800

    return (key, input_shape), (train_images, train_labels), (test_images, test_labels), (init_random_params, batch_size, total_number, predict, optimizer_init, optimizer_state, optimizer_update, get_params)

def train():

    (key, input_shape), (train_images, train_labels), (test_images, test_labels), (init_random_params, batch_size, total_number, predict, optimizer_init, optimizer_state, optimizer_update, get_params) = setup()
    epoch = int(total_number // batch_size)

    for i in range(20):

        print(f"Training epoch {i} is started")

        for j in range(epoch):

            start = i * batch_size
            end = (i + 1) * batch_size

            images = train_images[start: end]
            targets = train_labels[start: end]

            optimizer_state = update(j, optimizer_state, (images, targets), predict, get_params, optimizer_update)

            if (j + 1) % 10 == 0:

                params = get_params(optimizer_state)
                losses = loss_function(params, (images, targets), predict)

                print(f"Loss is {losses}")

        params = get_params(optimizer_state)

        print(f"Training epoch {epoch} is completed")

        accuracies = []
        correct_predictions = 0.0

        for j in range(epoch):

            start = i * batch_size
            end = (i + 1) * batch_size

            images = test_images[start: end]
            targets = test_labels[start: end]

            correct_predictions += verify_accuracy(params, (images, targets))

        accuracies.append(correct_predictions / float(total_number))

        print(f"Training accuracy is {accuracies}")

def main():

    train()

if __name__ == '__main__':

    main()


