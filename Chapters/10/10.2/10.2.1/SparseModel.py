import jax.random
import jax.experimental.sparse


def setup():

    key = jax.random.PRNGKey(17)
    number_classes = 10
    classes = jax.numpy.arange(number_classes)

    inputs = []
    genuines = []

    for i in range(1024):

        # Create item randomly
        item = jax.random.choice(key + 1, classes, shape = (1, ))[0]

        # One-hoted the item
        item_one_hot = jax.nn.one_hot(item, num_classes = number_classes)

        inputs.append(item_one_hot)
        genuines.append(item)

    # Generate the parameters for model
    parameters = [jax.random.normal(key, shape = (number_classes, 1)), jax.random.normal(key, shape = (1, ))]

    # Sparsify the inputs into sparse matrix
    sparsed_inputs = jax.experimental.sparse.BCOO.fromdense(jax.numpy.array(inputs))
    genuines = jax.numpy.array(genuines)

    epochs = 5000

    return sparsed_inputs, genuines, parameters, number_classes, classes, epochs

# Create the sigmoid function
def sigmoid(inputs):

    return 0.5 * (jax.numpy.tanh(inputs / 2.) + 1.)

# Create the model of prediction
def predict(parameters, inputs):

    output = jax.numpy.dot(inputs, parameters[0]) + parameters[1]

    return sigmoid(output)

# Create the loss function
def loss_function(parameters, sparsed_inputs, genuines):

    sparsed_predict = jax.experimental.sparse.sparsify(predict)
    genuines_hat = sparsed_predict(parameters, sparsed_inputs)

    genuines_hat = genuines * jax.numpy.log(genuines_hat) + (1 - genuines) * jax.numpy.log(1 - genuines_hat)

    return -jax.numpy.mean(genuines_hat)

def optimzier(parameters, sparsed_inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(parameters, sparsed_inputs, genuines)

    parameters = [(parameter - gradient * learning_rate) for parameter, gradient in zip(parameters, gradients)]

    return parameters

def train(parameters, sparsed_inputs, genuines, epochs):

    loss = loss_function(parameters, sparsed_inputs, genuines)

    print("Initial loss =", loss)

    for i in range(epochs):

        parameters = optimzier(parameters, sparsed_inputs, genuines)

        if (i + 1) % 100 == 0:
            loss = loss_function(parameters, sparsed_inputs, genuines)
            print(f"Epochs {i + 1} is iterated, now the loss =", loss)

def start():

    sparsed_inputs, genuines, parameters, number_classes, classes, epochs = setup()

    print(sparsed_inputs, genuines, parameters, number_classes, classes)

    train(parameters, sparsed_inputs, genuines, epochs)

def main():

    start()

if __name__ == "__main__":

    main()
