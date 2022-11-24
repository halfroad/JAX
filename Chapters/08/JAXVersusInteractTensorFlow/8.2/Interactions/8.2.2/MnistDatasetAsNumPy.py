import tensorflow_datasets


def dataset_as_numpy():

    trains = tensorflow_datasets.load(name = "mnist", split = tensorflow_datasets.Split.TRAIN, data_dir = "../../../../../../Shares/Datasets/MNIST/")
    trains = trains.shuffle(1024).batch(128).repeat(5).prefetch(10)

    for example in tensorflow_datasets.as_numpy(trains):

        numpy_images, numpy_labels = example["image"], example["label"]

        print(numpy_images, numpy_labels )

if __name__ == '__main__':

    dataset_as_numpy()
