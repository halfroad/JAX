import numpy
import os.path
import pickle


def setup(root = "../../../../../../../Shares/Datasets/cifar-10-batches-py/"):

    def load(file_name: str):

        name = os.path.join(root, file_name)

        with open(name, "rb") as handle:

            data = pickle.load(handle, encoding = "latin1")

            return data

    batch1 = load("data_batch_1")
    batch2 = load("data_batch_2")
    batch3 = load("data_batch_3")
    batch4 = load("data_batch_4")
    batch5 = load("data_batch_5")

    train_images = []
    train_labels = []

    for batch in [batch1, batch2, batch3, batch4, batch5]:

        image = (batch["data"]) / 255.
        label = (batch["labels"])

        train_images.append(image)
        train_labels.append(label)

    train_images = numpy.concatenate(train_images)
    train_labels = numpy.concatenate(train_labels)

    batch = load("test_batch")

    test_images = []
    test_labels = []

    for data_ in [batch]:

        image = (data_["data"])
        label = (data_["labels"])

        test_images.append(image)
        test_labels.append(label)

    test_images = numpy.concatenate(test_images)
    test_labels = numpy.concatenate(test_labels)

    return train_images, train_labels, test_images, test_labels

def main():

    train_images, train_labels, test_images, test_labels = setup()

    print("train_images.shape = ", train_images.shape, ", train_labels.shape = ", train_labels.shape, ", test_images.shape = ",  test_images.shape, ", test_labels.shape = ", test_labels.shape)

if __name__ == '__main__':

    main()


