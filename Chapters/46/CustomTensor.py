import numpy
import etils

class ArrayType:
    
    def __getitem__(self, idx):
        
        return Any
    
f32 = ArrayType()

def test():
    
    array = numpy.ones(shape = (3, 17, 21))
    
    print(array.shape)
    
    array = numpy.arange(1071).reshape(3, 17, 21)
    
    print(array.shape)
    
    x: etils.array_types.f32[(2, 3)] = numpy.ones(shape = (2, 3), dtype = numpy.float32)
    y: etils.array_types.f32[(3, 5)] = numpy.ones(shape = (3, 5), dtype = numpy.float32)
    
    z: etils.array_types.f32[(2, 5)] = x.dot(y)
    
    w: etils.array_types.f32[(7, 1, 5)] = numpy.ones((7, 1, 5), dtype = numpy.float32)
    
    q: etils.array_types.f32[(7, 2, 5)] = z + w
    
    print(f"x.shape = {x.shape}, y.shape = {y.shape}, z.shape = {z.shape}, w.shape = {w.shape}, q.shape = {q.shape}")
    
    x: f32[(2, 3)] = numpy.ones(shape = (2, 3), dtype = numpy.float32)
    y: f32[(3, 5)] = numpy.ones(shape = (3, 5), dtype = numpy.float32)
    
    z: f32[(2, 5)] = x.dot(y)
    
    w: f32[(7, 1, 5)] = numpy.ones((7, 1, 5), dtype = numpy.float32)
    
    q: f32[(7, 2, 5)] = z + w
    
    print(f"x.shape = {x.shape}, y.shape = {y.shape}, z.shape = {z.shape}, w.shape = {w.shape}, q.shape = {q.shape}")
    
def main():
    
    test()
    
if __name__ == "__main__":
    
    main()