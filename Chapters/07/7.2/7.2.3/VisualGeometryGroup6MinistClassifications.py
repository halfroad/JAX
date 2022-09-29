import MnistDatasetsV1

def setup():

    train_images, train_labels, test_images, test_labels = MnistDatasetsV1.mnist()

    batch_size = 312
    image_channel_dimension = 1

    return (train_images, train_labels), (test_images, test_labels), (batch_size, image_channel_dimension)

def start():

    (train_images, train_labels), (test_images, test_labels), (batch_size, image_channel_dimension) = setup()

    print(train_images, train_labels)

def main():

    start()

if __name__ == "__main__":

    main()

