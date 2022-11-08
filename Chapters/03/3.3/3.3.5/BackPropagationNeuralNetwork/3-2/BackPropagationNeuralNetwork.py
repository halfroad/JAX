from AccessoryFunctions import make_matrix, rand, sigmoid, sigmoid_derivative

class BackPropagationNeuralNetwork:

    def __init__(self):

        self.input_number = 0
        self.hidden_number = 0
        self.output_number = 0

        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []

        self.input_weights = []
        self.output_weights = []

    def setup(self, input_number, hidden_number, output_number):

        self.input_number = input_number + 1
        self.hidden_number = hidden_number
        self.output_number = output_number

        self.input_cells = [1.0] * self.input_number
        self.hidden_cells = [1.0] * self.hidden_number
        self.output_cells = [1.0] * self.output_number

        self.input_weights = make_matrix(self.input_number, self.hidden_number)
        self.output_weights = make_matrix(self.hidden_number, self.output_number)

        # Random activate
        for input_ in range(self.input_number):

            for hidden in range(self.hidden_number):

                self.input_weights[input_][hidden] = rand(-0.2, 0.2)

        for hidden in range(self.hidden_number):

            for output in range(self.output_number):

                self.output_weights[hidden][output] = rand(-2.0, 2.0)

    def predict(self, inputs):

        for i in range(self.input_number - 1):

            self.input_cells[i] = inputs[i]

        for i in range(self.hidden_number):

            total = 0.0

            for j in range(self.input_number):

                total += self.input_cells[j] * self.input_weights[j][i]

            self.hidden_cells[i] = sigmoid(total)

        for i in range(self.output_number):

            total = 0.0

            for j in range(self.hidden_number):

                total += self.hidden_cells[j] * self.output_weights[j][i]

            self.output_cells[i] = sigmoid(total)

        return self.output_cells[:]

    def back_propagate(self, case, labels, learning_rate):

        self.predict(case)

        # Compute the error of output layer
        output_deltas = [0.0] * self.output_number

        for i in range(self.output_number):

            error = labels[i] - self.output_cells[i]
            output_deltas[i] = sigmoid_derivative(self.output_cells[i]) * error

        # Compute the error of hidden layers
        hidden_deltas = [0.0] * self.hidden_number

        for i in range(self.hidden_number):

            error = 0.0

            for j in range(self.output_number):

                error += output_deltas[j] * self.output_weights[i][j]

            hidden_deltas[i] = sigmoid_derivative(self.hidden_cells[i]) * error

        # Update the weights of output layer
        for i in range(self.hidden_number):

            for j in range(self.output_number):

                self.output_weights[i][j] += learning_rate * output_deltas[j] * self.hidden_cells[i]

        # Update the weights of hidden layers
        for i in range(self.input_number):

            for j in range(self.hidden_number):

                self.input_weights[i][j] += learning_rate * hidden_deltas[j] * self.input_cells[i]

        error = 0.0

        for i in range(len(labels)):

            error += 0.5 * (labels[i] - self.output_cells[i]) ** 2

        return error

    def train(self, cases, labels, limit = 100, learning_rate = 5e-2):

        for i in range(limit):

            error = 0.0

            for j in range(len(cases)):

                label = labels[j]
                case = cases[j]

                error += self.back_propagate(case, label, learning_rate)

        pass

    def test(self):

        cases = [

            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]

        labels = [
            [0],
            [1],
            [1],
            [0]
        ]

        self.setup(2, 5, 1)

        self.train(cases, labels, 10000, 5e-2)

        for case in cases:

            print(self.predict(case))

if __name__ == "__main__":

    neuralNetwork = BackPropagationNeuralNetwork()

    neuralNetwork.test()



