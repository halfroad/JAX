import tensorflow as tf

def setup():
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

    return (train_images, train_labels), (test_images, test_labels)

def train():
    
    (train_images, train_labels), (test_images, test_labels) = setup()
    
    print((train_images.shape, train_labels.shape), (test_images.shape, test_labels.shape))
    
def main():
    
    train()
    
if __name__ == "__main__":
    
    main()