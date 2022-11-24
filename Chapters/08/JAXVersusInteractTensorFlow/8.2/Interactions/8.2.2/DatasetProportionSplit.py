import tensorflow_datasets


def dataset_split():

    splits = ["train[:50%]", "train[50%:75%]", "train[25%:]"]
    (trains, validations, tests), metadata = tensorflow_datasets.load(name = "mnist", split = splits, with_info = True, as_supervised = True, data_dir = "../../../../../../Shares/Datasets/MNIST/")

    print((trains, validations, tests), metadata)

if __name__ == '__main__':

    dataset_split()
