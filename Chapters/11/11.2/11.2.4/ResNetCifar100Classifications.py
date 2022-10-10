import sys
import jax.numpy
from jax.example_libraries import optimizers

sys.path.append("../11.2.2/")
import ResNetResidualModel

sys.path.append("../11.2.1/11-2/")
from CIFAR100DatasetPreparation import get_cifar100_train_and_test_sets



def setup():

    train_images_set, train_labels_set, test_images_set, test_labels_set = get_cifar100_train_and_test_sets("../../../../Exclusion/Datasets/cifar-10-batches-py/")
    train_images_set = jax.numpy.reshape(train_images_set, [-1, 3, 32, 32])
    train_images = jax.numpy.transpose(train_images_set, [0, 2, 3, 1])

    number_classes = 100

    train_labels = jax.nn.one_hot(train_labels_set, num_classes = number_classes)

    # test_images = jax.numpy.transpose(test_images_set, [0, 2, 3, 1])
    test_images = jax.numpy.transpose(test_images_set, [0, 1])
    test_labels = jax.nn.one_hot(test_labels_set, num_classes = number_classes)

    init_random_parameters, predict = ResNetResidualModel.ResNet50(number_classes)

    key = jax.random.PRNGKey(17)
    input_shape = [-1, 32, 32, 3]

    return train_images, train_labels, test_images, test_labels, number_classes, init_random_parameters, predict, key, input_shape

# Function to compute the accuracy
def compute_accuracy(parameters, batch, predict):

    """

    Correct the predictions over a minibatch

    """

    inputs, targets = batch

    result = predict(parameters, inputs)
    class_ = jax.numpy.argmax(result, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)

    return jax.numpy.sum(class_ == targets)

def loss(parameters, batch, predict):

    inputs, targets = batch

    losses = -targets * predict(parameters, inputs)
    losses = jax.numpy.sum(losses, axis = 1)

    return jax.numpy.mean(losses)

# Update the parameters of model
def update(i, opt_state, batch, get_parameters, opt_update, predict):

    """

    single optimization step over a minibatch

    """

    parameters = get_parameters(opt_state)
    grad_loss = jax.grad(loss)

    return opt_update(i, grad_loss(parameters, batch, predict), opt_state)

def train():

    train_images, train_labels, test_images, test_labels, number_classes, init_random_parameters, predict, key, input_shape = setup()

    # Here the step_size is learning_rate
    opt_init, opt_update, get_parameters = optimizers.adam(step_size = 2e-4)
    _, init_parameters = init_random_parameters(key, input_shape)
    opt_state = opt_init(init_parameters)

    # Can be adapted according to the hardware strength
    batch_size = 128

    # Set the number of train images accordingly
    total_number = 12800

    for i in range(17):

        epochs = int(total_number // batch_size)

        print(f"The train epoch number {i + 1} is started")

        for j in range(epochs):

            start = j * batch_size
            end = (i + 1) * batch_size

            images = train_images[start: end]
            labels = train_labels[start: end]

            opt_state = update(j, opt_state, (images, labels), get_parameters, opt_update, predict)

            if (j + 1) % 20 == 0:

                parameters = get_parameters(opt_state)
                loss_ = loss(parameters, (images, labels), predict)

                print(f"Loss: {loss_}")

        parameters = get_parameters(opt_state)

        print("Train is completed")

        accuracies = []
        correct_predictions = .0

        for j in range(epochs):

            start = i * batch_size
            end = (i + 1) * batch_size

            images = test_images[start: end]
            labels = test_labels[start: end]

            correct_predictions += compute_accuracy(parameters, (images, labels))

        accuracies.append(correct_predictions / float(total_number))

        print(f"Train set accuracy: {accuracies}")

def main():

    train()

if __name__ == "__main__":

    main()
