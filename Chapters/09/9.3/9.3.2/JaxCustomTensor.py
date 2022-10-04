import etils
import numpy
import sys
sys.path.append("9-11")

from F32CustomArrayType import f32


def start():

    array = numpy.ones(shape = (2, 17, 21))
    print(array)

    array = numpy.arange(714).reshape((2, 17, 21))
    print(array)

    """
    
    x: etils.array_types.f32[(2, 3)] = numpy.ones((2, 3), dtype = numpy.float32)
    y: etils.array_types.f32[(3, 5)] = numpy.ones((3, 5), dtype = numpy.float32)
    z: etils.array_types.f32[(2, 5)] = x.dot(y)
    w: etils.array_types.f32[(7, 1, 5)] = numpy.ones((7, 1, 5), dtype = numpy.float32)
    q: etils.array_types.f32[(7, 2, 5)] = z + w
    
    """
    x: f32[(2, 3)] = numpy.ones((2, 3), dtype = numpy.float32)
    y: f32[(3, 5)] = numpy.ones((3, 5), dtype = numpy.float32)
    z: f32[(2, 5)] = x.dot(y)
    w: f32[(7, 1, 5)] = numpy.ones((7, 1, 5), dtype = numpy.float32)
    q: f32[(7, 2, 5)] = z + w

    print("x = {}\ny = {}\nz = {}\nw = {}\nq = {}.".format(x, y, z, w, q))


def main():

    start()

if __name__ == "__main__":

    main()
