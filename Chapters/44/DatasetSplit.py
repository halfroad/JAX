import tensorflow_datasets as tfds

def setup():
    
    splits = ["train[:50%]", "train[:20%]", "train[:25%]"]
    
    (trains, validations, tests), metas = tfds.load(name = "mnist", data_dir = "/tmp/", split = list(splits), with_info = True, as_supervised = True)
    
    print(f"trains = {trains}, validations = {validations}, tests = {tests}), metas = {metas}")
    
def main():
    
    setup()
    
if __name__ == "__main__":
    
    main()