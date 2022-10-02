import sys
import time

sys.path.append("../8-4/")
from JaxMnistDatasetPreparation import setup
sys.path.append("../8-5/")
from JaxMnistModel import init_parameters
sys.path.append("../8-6/")
from JaxMnistModelComponents import loss_function, optimizer, accuracy_checkup

def train(train_images, train_labels, test_images, test_labels, number_pixels, output_dimensions):

    parameters = init_parameters(number_pixels, output_dimensions)

    begin = time.time()

    for i in range(500):

        parameters = optimizer(parameters, train_images, train_labels)

        if (i + 1) % 50 == 0:

            loss = loss_function(parameters, test_images, test_labels)
            end = time.time()
            accuracy = accuracy_checkup(parameters, test_images, test_labels) / float(1000.)

            print("%.12f is consumed" % (end - begin), f"while iterating {i}, the loss now is {loss}, accuracy of test set is {accuracy}")

            begin = time.time()

def main():

    (train_images, train_labels), (test_images, test_labels), number_pixels, output_dimensions = setup()
    train(train_images, train_labels, test_images, test_labels, number_pixels, output_dimensions)

if __name__ == "__main__":

    main()
