from Utils import create_matrix, rand
from ActivationFunctions import sigmoid, sigmoid_derivative


class BackwardsPropagationFeedbackNeuralNetwork:
    
    def  __init__(self):
        
        self.input_layer_cells_number = 0
        self.hidden_layer_cells_number = 0
        self.output_layer_cells_number = 0
        
        self.input_layer_cells = []
        self.hidden_layer_cells = []
        self.output_layer_cells = []
        
        self.input_layer_weights = []
        self.output_layer_weights = []
        
    def setup(self, input_layer_cells_number, hidden_layer_cells_number, output_layer_cells_number):
        
        self.input_layer_cells_number = input_layer_cells_number + 1
        self.hidden_layer_cells_number = hidden_layer_cells_number
        self.output_layer_cells_number = output_layer_cells_number
        
        self.input_layer_cells = [1.] * self.input_layer_cells_number
        self.hidden_layer_cells = [1.] * self.hidden_layer_cells_number
        self.output_layer_cells = [1.] * self.output_layer_cells_number
        
        self.input_layer_weights = create_matrix(self.input_layer_cells_number, self.hidden_layer_cells_number)
        self.output_layer_weights = create_matrix(self.hidden_layer_cells_number, self.output_layer_cells_number)
        
        for i in range(self.input_layer_cells_number):
            
            for h in range(self.hidden_layer_cells_number):
                
                self.input_layer_weights[i][h] = rand(-0.2, 0.2)
                
        for h in range(self.hidden_layer_cells_number):
            
            for o in range(self.output_layer_cells_number):
                
                self.output_layer_weights[h][o] = rand(-0.2, 0.2)
        
    def predict(self, inputs):
        
        for i in range(self.input_layer_cells_number - 1):
            
            self.input_layer_cells[i] = inputs[i]
            
        for i in range(self.hidden_layer_cells_number):
            
            total = 0.
            
            for j in range(self.input_layer_cells_number - 1):
                
                total += self.input_layer_cells[j] * self.input_layer_weights[j][i]
                
            self.hidden_layer_cells[i] = sigmoid(total)
            
        for i in range(self.output_layer_cells_number):
            
            total = 0.
            
            for j in range(self.hidden_layer_cells_number):
                
                total += self.hidden_layer_cells[j] * self.output_layer_weights[j][i]
                
            self.output_layer_cells[i] = sigmoid(total)
            
        return self.output_layer_cells[:]
    
    def back_propagate(self, case, label, learning_rate):
        
        self.predict(case)
        
        # Compute the error of output layer
        output_layer_deltas = [0.] * self.output_layer_cells_number
        
        for i in range(self.output_layer_cells_number):
            
            error = label[i] - self.output_layer_cells[i]
            output_layer_deltas[i] = sigmoid_derivative(self.output_layer_cells[i]) * error
            
        # Compute the error of hidden layer
        hidden_layer_deltas = [0.0] * self.hidden_layer_cells_number
        
        for i in range(self.hidden_layer_cells_number):
            
            error = 0.0
            
            for j in range(self.output_layer_cells_number):
                
                error += output_layer_deltas[j] * self.output_layer_weights[i][j]
                hidden_layer_deltas[i] = sigmoid_derivative(self.hidden_layer_cells[i]) * error
                
        # Update the weights of output layer
        for i in range(self.hidden_layer_cells_number):
            
            for j in range(self.output_layer_cells_number):
                
                self.output_layer_weights[i][j] += learning_rate * output_layer_deltas[j] * self.hidden_layer_cells[i]
                
        # Update the weights of hidden layer
        for i in range(self.input_layer_cells_number):
            
            for j in range(self.hidden_layer_cells_number):
                
                self.input_layer_weights[i][j] += learning_rate * hidden_layer_deltas[j] * self.input_layer_cells[i]
                
        error = 0.0
        
        for i in range(len(label)):
            
            error += 0.5 * (label[i] - self.output_layer_cells[i]) ** 2
            
        return error
    
    def train(self, cases, labels, limit = 100, learning_rate = 5e-2):
        
        for i in range(limit):
            
            error = 0.0
            
            for j in range(len(cases)):
                
                label = labels[j]
                case = cases[j]
                
                error += self.back_propagate(case,label, learning_rate)
                
            print(f"Error now is {error}, epochs {i + 1}")
                
        pass
    
    
    def test(self):
        
        cases = [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]]
        labels = [[0], [1], [1], [0]]
        
        self.setup(2, 5, 1)
        
        self.train(cases, labels, 10000, 5e-2)
        
        for case in cases:
            
            print(self.predict(case))
            
if __name__ == "__main__":
    
    network = BackwardsPropagationFeedbackNeuralNetwork()
    network.test()