import pickle


def load(name):

    with open(name, "rb") as handler:

        batch = pickle.load(handler, encoding = "latin1")

        return batch

def start():

    batch1 = load("../../../../Exclusion/Datasets/cifar-10-batches-py/data_batch_1")
    print(batch1.keys())
    print(batch1)

def main():

    start()

if __name__ == "__main__":

    main()
