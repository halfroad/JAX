import jax

# Create the sigmoid function
def sigmoid(inputs):

    return 0.5 * (jax.numpy.tanh(inputs / 2.) + 1.)

# Create the model of prediction
def predict(parameters, inputs):

    output = jax.numpy.dot(inputs, parameters[0]) + parameters[1]

    return sigmoid(output)

# Create the loss function
def loss_function(parameters, sparsed_inputs, genuines):

    sparsed_predict = jax.experimental.sprase.sparsify(predict)
    genuines_hat = sparsed_predict(parameters, sparsed_inputs)

    genuines_hat = genuines * jax.numpy.log(genuines_hat) + (1 - genuines) * jax.numpy.log(1 - genuines_hat)

    return -jax.numpy.mean(genuines_hat)

def optimzier(parameters, sparsed_inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(parameters, sparsed_inputs, genuines)

    parameters = [(parameter - gradient * learning_rate) for parameter, gradient in zip(parameters, gradients)]

    return parameters

def train(parameters, sparsed_inputs, genuines):

    loss = loss_function(parameters, sparsed_inputs, genuines)

    print("Initial loss =", loss)

    for i in range(100):

        parameters = optimzier(parameters, sparsed_inputs, genuines)
        loss = loss_function(parameters, sparsed_inputs, genuines)

        print("loss =", loss)
