import pickle

def setup():
    
    def load(fileName: str):
        
        with open(file = fileName, mode = "rb") as handler:
            
            data = pickle.load(file = handler, encoding = "latin1")
            
        return data
    
    trains = load("../../Shares/cifar-100-python/train")
    tests = load("../../Shares/cifar-100-python/test")
    metas = load("../../Shares/cifar-100-python/meta")
    
    return trains, tests, metas
    
def train():
    
    trains, tests, metas = setup()
    
    for key in trains.keys():
        
        print(f"key = {key}, len(trains[key]) = {len(trains[key])}")
    
    print("--------------------------------------------------")
    
    print(trains["batch_label"])
    
    for key in tests.keys():
        
        print(f"key = {key}, len(tests[key]) = {len(tests[key])}")
    
    print("--------------------------------------------------")
    
    for key in metas.keys():
        
        print(f"key = {key}, len(metas[key]) = {len(metas[key])}")
    
def main():
    
    train()
    
if __name__ == "__main__":
    
    main()