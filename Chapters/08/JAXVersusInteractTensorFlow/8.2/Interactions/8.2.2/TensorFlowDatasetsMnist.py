import tensorflow
import tensorflow_datasets


def show():

    mnist = tensorflow_datasets.load(name = "mnist", data_dir = "../../../../../../Shares/Datasets/MNIST/")

    trains, tests = mnist["train"], mnist["test"]

    print(trains, tests)


if __name__ == '__main__':

    show()
