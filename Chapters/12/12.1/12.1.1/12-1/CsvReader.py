import csv
import sys


def read(path):

    csv.field_size_limit(sys.maxsize)

    with open(path, encoding = "latin1", mode = "r") as handle:

        news = csv.reader(handle)

        labels = []
        titles = []
        texts = []

        for line in news:

            labels.append(line[0])
            titles.append(line[1].lower())
            texts.append(line[2].lower())

        return labels, titles, texts

def start():

    labels, titles, texts = read("../../../../../Exclusion/Datasets/newsSpace")

    print(labels.shape, titles.shape, texts.shape)

def main():

    start()

if __name__ == "__main__":

    main()
