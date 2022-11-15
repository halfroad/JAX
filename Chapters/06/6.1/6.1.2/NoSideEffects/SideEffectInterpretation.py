import numpy


def in_place_modify(x):

    x[0] = 123

    return None

def start():

    x = numpy.array([1, 2, 3])

    in_place_modify(x)

    print(x)

if __name__ == '__main__':

    start()
