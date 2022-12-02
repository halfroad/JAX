import pickle


def read(file_name: str):

    with open(file_name, "rb") as handle:

        cifar = pickle.load(handle, encoding = "latin1")

        return cifar

def main():

    root = "../../../../../../Shares/Datasets/cifar-10-batches-py/"
    file_name = root + "data_batch_1"

    cifar = read(file_name)

    print(cifar.keys())

if __name__ == '__main__':

    main()

