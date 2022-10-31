"""

Paragraph 2.2.3, Model Design based on JAX Linear Regression
Page 26
Third Step: Train the Model

"""
import jax.numpy
import sklearn.datasets


def Dense(shape = [4, 1]):

    def init_function(input_shape = shape):

        prng = jax.random.PRNGKey(15)

        weights, biases = jax.random.normal(prng, shape = input_shape), jax.random.normal(prng, shape = (input_shape[-1],))

        return weights, biases

    def apply_function(inputs, parameters):

        weight, bias = parameters

        return jax.numpy.dot(inputs, weight) + bias

    return init_function, apply_function

def linear_loss_function(parameters, inputs, genuines, apply_function):

    """

    Loss Function

    g(x) = (f(x) - y)²

    """

    predictions = apply_function(inputs, parameters)
    powered = jax.numpy.power(predictions - genuines, 2.0)
    mse = jax.numpy.mean(powered)

    return mse

def setup():

    iris = sklearn.datasets.load_iris()

    data = iris.data
    targets = iris.target

    data = jax.numpy.float32(data)
    targets = jax.numpy.float32(targets)

    # The number of iterations for gradent
    epochs = 1000

    # Learning Rate or Step Size
    learning_rate = 5e-3

    init_function, apply_function = Dense()
    initial_parameters = init_function()

    return epochs, learning_rate, init_function, apply_function, initial_parameters, data, targets

def train():

    epochs, learning_rate, init_function, apply_function, parameters, data, targets = setup()

    for i in range(epochs):

        # Compute the loss
        loss = linear_loss_function(parameters, data, targets, apply_function)

        if (i + 1) % 100 == 0:

            print(f"Iteration: {i + 1}, the loss now is {loss}")

        # Compute the gradient and update the parameters
        grad_linear_loss_function = jax.grad(linear_loss_function)
        gradients = grad_linear_loss_function(parameters, data, targets, apply_function)

        parameters = [
            # Plus each parameter by the value of learning_rate * (-) gradient
            (parameter - gradient * learning_rate) for parameter, gradient in zip(parameters, gradients)
        ]

    print(f"Iteration: {i + 1}, the final loss is {loss}")

if __name__ == "__main__":

    train()
