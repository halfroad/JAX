import math
import random


def rand(a, b):

    """

    Get random number from lower a to upper b

    """
    return (b - a) * random.random() + a


def built_matrix(inputs_number, outputs_number, fill = 0.0):

    matrix = []

    for i in range(inputs_number):
        matrix.append([fill] * outputs_number)

    return matrix


def sigmoid(x):

    sig = 1.0 / (1 + math.exp(-x))

    return sig


def sigmoid_differentiate(x):

    derivative = x * (1 - x)

    return derivative


class BackPropagationNeuralNetworks:

    def __init__(self):

        self.inputs_number = 0
        self.hiddens_number = 0
        self.outputs_number = 0

        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []

        self.input_weights = []
        self.output_weights = []

    def configure(self, inputs_number, hiddens_number, outputs_number):

        self.inputs_number = inputs_number + 1
        self.hiddens_number = hiddens_number
        self.outputs_number = outputs_number

        self.input_cells = [1.0] * self.inputs_number
        self.hidden_cells = [1.0] * self.hiddens_number
        self.output_cells = [1.0] * self.outputs_number

        self.input_weights = built_matrix(self.inputs_number, self.hiddens_number)
        self.output_weights = built_matrix(self.hiddens_number, self.outputs_number)

        # Random activate
        for i in range(self.inputs_number):

            for j in range(self.hiddens_number):

                self.input_weights[i][j] = rand(-.2, .2)

        for i in range(hiddens_number):

            for j in range(self.outputs_number):

                self.output_weights[i][j] = rand(-2.0, 2.0)

    def predict(self, inputs):

        for i in range(self.inputs_number - 1):

            self.input_cells[i] = inputs[i]

        for i in range(self.hiddens_number):

            total = 0.0

            for j in range(self.inputs_number):

                total += self.input_cells[j] * self.input_weights[j][i]

            self.hidden_cells[i] = sigmoid(total)

        for i in range(self.outputs_number):

            total = 0.0

            for j in range(self.hiddens_number):

                total += self.hidden_cells[j] * self.output_weights[j][i]

            self.output_cells[i] = sigmoid(total)

        return self.output_cells[:]

    def back_propagate(self, case, label, learn):

        self.predict(case)

        # Compute the deltas of output layer
        output_deltas = [0.0] * self.outputs_number

        for i in range(self.outputs_number):

            error = label[i] - self.output_cells[i]
            output_deltas[i] = sigmoid_differentiate(self.output_cells[i]) * error

        # Compute the deltas of hidden layer
        hidden_deltas = [0.0] * self.hiddens_number

        for i in range(self.hiddens_number):

            error = 0.0

            for j in range(self.outputs_number):

                error += output_deltas[j] * self.output_weights[i][j]

            hidden_deltas[i] = sigmoid_differentiate(self.hidden_cells[i]) * error

        # Update the weights of output layer
        for i in range(self.hiddens_number):

            for j in range(self.outputs_number):

                self.output_weights[i][j] += learn * output_deltas[j] * self.hidden_cells[i]

        # Update the weights of hidden layer
        for i in range(self.inputs_number):

            for j in range(self.hiddens_number):

                self.input_weights[i][j] += learn * hidden_deltas[j] * self.input_cells[i]

        error = 0

        for i in range(len(label)):

            error += 0.5 * (label[i] - self.output_cells[i]) ** 2

        return error

    def train(self, cases, labels, limit = 100, learn = 0.05):

        for _ in range(limit):

            error = 0.0

            for i in range(len(cases)):

                label = labels[i]
                case = cases[i]

                error += self.back_propagate(case, label, learn)

        pass

    def start(self):

        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        labels = [
            [0],
            [1],
            [1],
            [0]
        ]

        self.configure(2, 5, 1)
        self.train(cases, labels, 10000, 0.05)

        for case in cases:

            print(self.predict(case))

if __name__ == "__main__":

    nn = BackPropagationNeuralNetworks()

    nn.start()


