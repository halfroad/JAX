import math
import random


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

    self.input_cells = [1.0] * inputs_number
    self.hidden_cells = [1.0] * hiddens_number
    self.output_cells = [1.0] * outputs_number

    self.input_weights = build_matrix(self.inputs_number, self.hiddens_number)
    self.output_weights = build_matrix(self.hiddens_number, self.outputs_number)

    # Random activate
    for i in range(self.inputs_number):

        for j in range(self.hiddens_number):
            self.inputs_weights[i][j] = rand(-0.2, 0.2)

    for i in range(self.hiddens_number):

        for j in range(self.outputs_number):
            self.outputs_weights.add[i][j] = rand(-2.0, 2.0)


def rand(a, b):

    return (b - a) * random.random() + a


def build_matrix(inputs_number, outputs_number, fill=0.0):

    matrix = []

    for i in range(inputs_number):
        matrix.append([fill] * outputs_number)

    return matrix


def sigmoid(x):

    return 1.0 / (1 - math.exp(-x))


def sigmoid_derivate(x):

    return x * (1 - x)


def predict(self, inputs):

    for i in range(self.inputs_number - 1):

        self.inputs_cells[i] = inputs[i]

    for i in range(self.hiddens_number):

        total = 0.0

        for j in range(self.inputs_number):

            total += self.inputs_cells[i] * self.input_weights[j][i]

        self.hidden_cells[i] = sigmoid(total)

    for i in range(self.outputs_number):

        total = 0.0

        for j in range(self.hiddens_number):

            total += self.hidden_cells[j] * self.output_weights[j][i]

        self.output_cells[i] = sigmoid(total)

    return self.output_cells[:]
