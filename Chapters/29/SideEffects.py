import numpy

x = numpy.array([1, 2, 3])

def update_in_place(x):
    
    x[0] = 10
    
    return None

def test():
    
    update_in_place(x)
    
    print("x = ", x)
    
if __name__ == "__main__":
    
    test()