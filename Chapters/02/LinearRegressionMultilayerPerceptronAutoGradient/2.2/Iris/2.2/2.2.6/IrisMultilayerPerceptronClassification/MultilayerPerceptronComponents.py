import jax.random

"""

Paragraph 2.2.6, Experiment on Iris Classification based Multilayer Perceptron
Page 32
Second Step: Multilayer Perceptron Components

"""

# Fully Connected Layer
def Dense(shape = [1, 1]):

    def init_function(input_shape = shape):

        prng = jax.random.PRNGKey(15)

        weights, biases = jax.random.normal(prng, shape = input_shape), jax.random.normal(prng, shape = (input_shape[-1]))

        return weights, biases

    def apply_function(inputs, parameters):

        weight, bias = parameters

        return jax.numpy.dot(inputs, weight) + bias

    return init_function, apply_function

# Activation Function
def tanh(inputs):

    return jax.numpy.tanh(inputs)

def relu(inputs, alpha = 1.67, theta = 1.05):

    return jax.numpy.where(inputs > 0, inputs, alpha * (jax.numpy.exp(inputs) - 1)) * theta

# Softmax Function
def softmax(inputs, axis = -1):

    unnormalized = jax.numpy.exp(inputs)

    return unnormalized / unnormalized.sum(axis, keepdims = True)

# Cross Entropy
def cross_entropy(genuines, predictions, delta = 1e-7):

    predictions = predictions + delta
    logs = jax.numpy.log(predictions)
    crosses = genuines * logs

    entropys = -jax.numpy.sum(crosses, axis = -1)

    return entropys
