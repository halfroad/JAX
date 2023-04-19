import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

def setup():
    
    mnist = tfds.load(name = "mnist", data_dir = "/tmp")
    trains, tests = mnist["train"], mnist["test"]
    
    assert isinstance(trains, tf.data.Dataset)
    
    print(trains, tests)
    
def main():
    
    setup()
    
if __name__ == "__main__":
    
    main()