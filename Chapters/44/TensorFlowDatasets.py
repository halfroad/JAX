import tensorflow_datasets as tfds

def preview_datasets():
    
    builders = tfds.list_builders()
    
    print("Length of builders = ", len(builders))
    print("builders = ", builders)
    
if __name__ == "__main__":
    
    preview_datasets()