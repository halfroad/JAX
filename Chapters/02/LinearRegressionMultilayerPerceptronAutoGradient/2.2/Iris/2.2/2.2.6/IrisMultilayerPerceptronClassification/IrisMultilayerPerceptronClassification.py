"""

Paragraph 2.2.6, Experiment on Iris Classification based Multilayer Perceptron
Page 32
Forth Step: Classify the Iris with Multilayer Perceptron
Program 2-6

"""

import jax

from IrisDatasetPreparation import setup
from MultilayerPerceptronModel import mlp, mlp_loss_function


def train():

    # Since there are 2 linear layers, 5 parameters are needed to inject into the model as the initial parameters

    data, targets, prng, epochs, parameters, learning_rate = setup()

    for i in range(epochs):

        losses = mlp_loss_function(parameters, data, targets)

        if (i + 1) % 1000 == 0:

            predictions = mlp(data, parameters)

            # argmax() is to figure out the maximim x when the f(x) has the peak
            class_ = jax.numpy.argmax(predictions, axis = 1)
            target_ = jax.numpy.argmax(targets, axis = 1)

            accuracy = jax.numpy.sum(class_ == target_) / len(target_)

            print(f"Iteration: {i + 1}, loss = {losses}, accuracy = {accuracy}")

        grad_mlp_loss_function = jax.grad(mlp_loss_function)
        gradients = grad_mlp_loss_function(parameters, data, targets)

        parameters = [
            (parameter - gradient * learning_rate) for parameter, gradient in zip(parameters, gradients)
        ]

    predictions = mlp(data, parameters)

    class_ = jax.numpy.argmax(predictions, axis = 1)
    target_ = jax.numpy.argmax(targets, axis = 1)

    accuracy = jax.numpy.sum(class_ == target_) / len(target_)

    print(f"Final accuracy is {accuracy}")

if __name__ == "__main__":

    train()
