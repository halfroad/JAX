import csv


def setup():

    handle = open("../../../../../../../Shares/Datasets/agnews/train.csv", mode = "r")
    trains = csv.reader(handle)

    return trains

def split():

    labels = []
    titles = []
    descriptions = []

    trains = setup()

    for line in trains:

        labels.append(line[0])
        titles.append(line[1].lower())
        descriptions.append(line[2].lower())

    print("labels = ", labels, "titles = ", titles, "descriptions = ", descriptions)

def main():

    split()

if __name__ == '__main__':

    main()
