import jax.numpy
import tensorflow_datasets


def prepare():

    trains = tensorflow_datasets.load(name = "mnist", split = tensorflow_datasets.Split.TRAIN, batch_size = -1, data_dir = "../../../../../../Shares/Datasets/MNIST/")

    trains = tensorflow_datasets.as_numpy(trains)
    images, labels = trains["image"], trains["label"]

    # images = jax.numpy.expand_dims(images, axis = -1)
    images = (images - 256) / 256.

    return images, labels

def train():

    images, labels = prepare()

    print(images.shape, labels.shape)

if __name__ == '__main__':

    train()
