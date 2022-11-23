import time


def train(params, optimizer, loss_function, verify_accuracy, train_images, train_labels, test_images, test_labels):

    start = time.time()

    for i in range(500):

        params = optimizer(params, train_images, train_labels)

        if (i + 1) % 100 == 0:

            loss = loss_function(params, test_images, test_labels)
            accuracy = verify_accuracy(params, test_images, test_labels) / float(10000.0)

            end = time.time()

            print(f"%.12fs" % (end - start), f"is consumed while iterating {i + 1} epochs, the loss now is {loss}, the accuracy is {accuracy}")

            start = time.time()
