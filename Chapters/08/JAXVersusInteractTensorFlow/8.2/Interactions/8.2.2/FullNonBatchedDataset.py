import tensorflow_datasets


def dataset_non_batched():

    trains = tensorflow_datasets.load(name = "mnist", split = tensorflow_datasets.Split.TRAIN, batch_size = -1, data_dir = "../../../../../../Shares/Datasets/MNIST/")
    numpy_trains = tensorflow_datasets.as_numpy(trains)

    numpy_images, numpy_labels = numpy_trains["image"], numpy_trains["label"]

    print(numpy_images.shape, numpy_labels.shape)

if __name__ == '__main__':

    dataset_non_batched()
