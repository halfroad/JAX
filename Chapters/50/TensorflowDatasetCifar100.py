import tensorflow_datasets as tfds

def setup():
    
    (trains, tests), meta = tfds.load("cifar100", data_dir = "/tmp/", split = [tfds.Split.TRAIN, tfds.Split.TEST], with_info = True, batch_size = -1)
    
    #tensorflow_datasets.show_examples(trains, metas)
        
    trains = tfds.as_numpy(trains)
    tests = tfds.as_numpy(tests)
    
    train_images, train_labels = trains["image"], trains["label"]
    test_images, test_labels = tests["image"], tests["label"]
    
    return (train_images, train_labels), (test_images, test_labels)

def train():

    (train_images, train_labels), (test_images, test_labels) = setup()
    
    print((train_images.shape, train_labels.shape), (test_images.shape, test_labels.shape))
    
def main():
    
    train()
    
if __name__ == "__main__":
    
    main()