import numpy as np

def dimension1():

    a = np.linspace(1, 3, 3, dtype = np.int32)
    b = np.linspace(2, 4, 3, dtype = np.int32)

    print("a = {}, a.shape = {}".format(a, a.shape))
    print("b = {}, b.shape = {}".format(b, b.shape))
    print("-------------------------------")

    stacked = np.stack((a, b), axis = 0)
    print("np.stack((a, b), axis = 0) = {},\n np.stack((a, b), axis = 0).shape = {}".format(stacked, stacked.shape))
    print("-------------------------------")

    stacked = np.stack((a, b), axis = 1)
    print("np.stack((a, b), axis = 1) = {},\n np.stack((a, b), axis = 1).shape = {}".format(stacked, stacked.shape))
    print("-------------------------------")

    c = np.linspace(3, 5, 3, dtype = np.int32)

    print("c = {}, c.shape = {}".format(c, c.shape))

    stacked = np.stack((a, b, c), axis = 0)
    print("np.stack((a, b, c), axis = 0) = {},\n np.stack((a, b, c), axis = 0).shape = {}".format(stacked, stacked.shape))
    print("-------------------------------")

    stacked = np.stack((a, b, c), axis = 1)
    print("np.stack((a, b, c), axis = 1) = {},\n np.stack((a, b, c), axis = 1).shape = {}".format(stacked, stacked.shape))
    print("-------------------------------")

def dimension2():

    array1 = np.arange(0, 12, 1).reshape(3, 4)
    array2 = np.arange(12, 24, 1).reshape(3, 4)
    array3 = np.arange(24, 36, 1).reshape(3, 4)

    arrays = (array1, array2, array3)

    print("(array1, array2, array3) = ", arrays)

    stacked0 = np.stack(arrays, axis = 0)

    print("np.stack((array1, array2, array3), axis = 0) = {}, np.stack((array1, array2, array3), axis = 0).shape = {}".format(stacked0, stacked0.shape))
    print("----------------------------------------------")

    stacked1 = np.stack((array1, array2, array3), axis = 1)

    print("np.stack((array1, array2, array3), axis = 1) = {}, np.stack((array1, array2, array3), axis = 1).shape = {}".format(stacked1, stacked1.shape))
    print("----------------------------------------------")

    stacked2 = np.stack((array1, array2, array3), axis = 2)

    print("np.stack((array1, array2, array3), axis = 2) = {}, np.stack((array1, array2, array3), axis = 2).shape = {}".format(stacked2, stacked2.shape))
    print("----------------------------------------------")

def main():

    dimension2()

if __name__ == "__main__":

    main()
