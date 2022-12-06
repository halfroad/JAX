import csv


def setup():

    handle = open("../../../../../../../Shares/Datasets/agnews/train.csv", mode = "r")
    trains = csv.reader(handle)

    for line in trains:

        print(line)

def main():

    setup()

if __name__ == '__main__':

    main()
