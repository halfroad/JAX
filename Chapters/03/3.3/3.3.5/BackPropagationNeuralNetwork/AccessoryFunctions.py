import math
import random



def rand(lower, upper):

    return (upper - lower) * random.random() + lower

def make_matrix(layers_number, next_layers_number, fill = 0.0):

    matrix = []

    for i in range(layers_number):

        matrix.append([fill] * next_layers_number)

    return matrix

def sigmoid(x):

    return 1.0 / (1.0 + math.exp(-1))

def sigmoid_derivative(x):

    return x * (1 - x)
