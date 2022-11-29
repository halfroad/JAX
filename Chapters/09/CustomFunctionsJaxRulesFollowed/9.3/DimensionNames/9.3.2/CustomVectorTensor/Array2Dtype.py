import numpy
import typing
from F32CustomArrayType import f32

def array_to_dtype():

    x: f32[(2, 3)] = numpy.ones(shape = (2, 3), dtype = numpy.float32)
    y: f32[(3, 5)] = numpy.ones(shape = (3, 5), dtype = numpy.float32)

    z: f32[(2, 5)] = x.dot(y)

    w: f32[(7, 1, 5)] = numpy.ones(shape = (7, 1, 5), dtype = numpy.float32)

    q: f32[(7, 2, 5)] = z + w

    print("x = ", x, "\ny = ", y, "\nz = ", z, "\nw = ", w, "\nq = ", q)

    print(q.shape)
if __name__ == '__main__':

    array_to_dtype()


