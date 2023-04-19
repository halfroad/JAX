import tensorflow as tf
import tensorflow_datasets as tfds

def setup():
    
    trains = tfds.load(name = "mnist", split = tfds.Split.TRAIN, data_dir = "/tmp")
    trains = tfds.load(name = "mnist", batch_size = -1, split = tfds.Split.TRAIN, data_dir = "/tmp")
    trains = trains.shuffle(1024).batch(128).repeat(5).prefetch(10)
    
    i = 0
    
    for item in tfds.as_numpy(trains):
        
        images, labels = item["image"], item["label"]
        
        print(f"i = {i}, images.shape = {images.shape}, labels.shape = {labels.shape}")
        
        i = i + 1
        
def setup_():
    
    trains = tfds.load(name = "mnist", batch_size = -1, split = tfds.Split.TRAIN, data_dir = "/tmp")
    trains = tfds.as_numpy(trains)
    
    train_images, train_labels = trains["image"], trains["label"]
    
    print(f"train_images.shape = {train_images.shape}, train_labels.shape = {train_labels.shape}")
        
        
def main():
    
    # setup()
    setup_()
    
if __name__ == "__main__":
    
    main()