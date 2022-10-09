import os
import pickle

import numpy


def load(file_name):

    with open(file_name, "rb") as handle:

        data = pickle.load(handle, encoding = "latin1")

    return data

def get_cifar100_train_images_and_labels(root = ""):

    batch1 = load(os.path.join(root, "data_batch_1"))
    batch2 = load(os.path.join(root, "data_batch_2"))
    batch3 = load(os.path.join(root, "data_batch_3"))
    batch4 = load(os.path.join(root, "data_batch_4"))
    batch5 = load(os.path.join(root, "data_batch_5"))

    train_images_set = []
    train_labels_set = []

    for item in [batch1, batch2, batch3, batch4, batch5]:

        image = item["data"] / 255.
        label = item["labels"]

        train_images_set.append(image)
        train_labels_set.append(label)

    train_images_set = numpy.concatenate(train_images_set)
    train_labels_set = numpy.concatenate(train_labels_set)

    return train_images_set, train_labels_set


def get_cifar100_test_images_and_labels(root = ""):

    batch = load(os.path.join(root, "test_batch"))

    test_images_set = []
    test_labels_set = []

    for item in [batch]:

        image = item["data"]
        label = item["labels"]

        test_images_set.append(image)
        test_labels_set.append(label)

    test_images_set = numpy.concatenate(test_images_set)
    test_labels_set = numpy.concatenate(test_labels_set)

    return test_images_set, test_labels_set

def get_cifar100_train_and_test_sets(root = ""):

    train_images_set, train_labels_set = get_cifar100_train_images_and_labels(root = root)
    test_images_set, test_labels_set = get_cifar100_test_images_and_labels(root = root)

    return train_images_set, train_labels_set, test_images_set, test_labels_set

def start():

    train_images_set, train_labels_set, test_images_set, test_labels_set = get_cifar100_train_and_test_sets("../../../../../Exclusion/Datasets/cifar-10-batches-py/")

    print(train_images_set.shape, train_labels_set.shape, test_images_set.shape, test_labels_set.shape)

def main():

    start()

if __name__ == "__main__":

    main()
