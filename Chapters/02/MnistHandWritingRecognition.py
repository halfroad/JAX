import tensorflow_datasets

def setup():
    
    dataset, metadata = tensorflow_datasets.load(name = "mnist", split = [tensorflow_datasets.Split.TRAIN, tensorflow_datasets.Split.TEST], batch_size =-1, as_supervised = True, with_info = True, data_dir = "../../Shares/Datasets/MNIST/")
    
    (train_images, train_labels), (test_images, test_labels) = tensorflow_datasets.as_numpy(dataset)
    
    return (train_images, train_labels), (test_images, test_labels)
    
def train():
    
    (train_images, train_labels), (test_images, test_labels) = setup()
    
    print((train_images, train_labels), (test_images, test_labels))
    
def main():
    
    train()
    
if __name__ == "__main__":
    
    main()