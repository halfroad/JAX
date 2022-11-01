import jax


"""

Paragraph 2.2.6, Experiment on Iris Classification based Multilayer Perceptron
Page 32
Third Step: Multilayer Perceptron Model Design

"""

from MultilayerPerceptronComponents import Dense, tanh, softmax, cross_entropy

def mlp(inputs, parameters):

    # Import the parameters
    a0, b0, a1, b1 = parameters
    _, apply_function = Dense()

    inputs = apply_function(inputs, [a0, b0])
    inputs = tanh(inputs)

    _, apply_function = Dense()

    inputs = apply_function(inputs, [a1, b1])
    inputs = softmax(inputs, axis = -1)

    return inputs

def mlp_loss_function(parameters, inputs, genuines):

    predictions = mlp(inputs, parameters)

    losses = cross_entropy(genuines, predictions)

    return jax.numpy.mean(losses)

