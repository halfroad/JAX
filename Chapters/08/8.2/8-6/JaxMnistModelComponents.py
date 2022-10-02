import jax.numpy

def forward(parameters, inputs):

    for parameter in parameters[: -1]:

        weight = parameter["weight"]
        bias = parameter["bias"]

        inputs = jax.numpy.dot(inputs, weight) + bias
        # Relu is defined in 8-6
        inputs = relu(inputs)

    output = jax.numpy.dot(inputs, parameters[-1]["weight"]) + parameters[-1]["bias"]

    print(output)

    output = jax.nn.softmax(output, axis = -1)

    return output

@jax.jit
def relu(input_):

    """

    Activation function

    """

    return jax.numpy.maximum(0, input_)

@jax.jit
def cross_entropy(genuines, predictions):

    """

    Cross-entropy function

    """
    entropys = genuines * jax.numpy.log(jax.numpy.clip(predictions, 1e-9, .999)) + (1 - genuines) * jax.numpy.log(jax.numpy.clip(1 - predictions, 1e-9, .999))
    entropys = -jax.numpy.sum(entropys)

    return jax.numpy.mean(entropys)

@jax.jit
def loss_function(parameters, inputs, genuines):

    """

    Loss function

    """

    predictions = forward(parameters, inputs)
    entropys = cross_entropy(genuines, predictions)

    return entropys

@jax.jit
def optimizer(parameters, inputs, genuines, learning_rate = 1e-3):

    """

    SGD optimizer function

    """
    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(parameters, inputs, genuines)

    parameters = jax.tree_util.tree_map(lambda parameter, gradient: parameter - gradient * learning_rate, parameters, gradients)

    return parameters

@jax.jit
def accuracy_checkup(parameters, inputs, targets):

    """

    Calculus of accuracy

    """

    result = forward(parameters, inputs)
    classification = jax.numpy.argmax(result, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)
    equals = jax.numpy.sum(classification == targets)

    return jax.numpy.sum(equals)
