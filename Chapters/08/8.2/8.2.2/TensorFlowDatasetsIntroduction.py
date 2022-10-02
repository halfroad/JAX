import tensorflow_datasets as tfds
import tensorflow as tf

def start():

    print(tfds.list_builders())

    mnist = tfds.load("mnist")

    trains, tests = mnist["train"], mnist["test"]

    assert isinstance(trains, tf.data.Dataset)

    print(trains, tests)

    trains = tfds.load("mnist", split = tfds.Split.TRAIN)
    trains = trains.shuffle(1024).batch(128).repeat(5).prefetch(10)

    for example in tfds.as_numpy(trains):

        images, labels = example["image"], example["label"]

        print(images, labels)

    trains = tfds.load("mnist", split = tfds.Split.TRAIN, batch_size = -1)
    trains = tfds.as_numpy(trains)

    images, labels = trains["image"], trains["label"]

    print(images, labels)

    splits = tfds.Split.TRAIN.subsplit(weighted = [2, 1, 1])

    (trains, validations, tests), metas = tfds.load("mnist", split = list(splits), with_info = True, as_supervised = True)

    print(images, validations, labels, metas)


def main():

    start()

if __name__ == "__main__":

    main()
