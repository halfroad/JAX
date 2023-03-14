import jax

def accumulate():
    
    # Error: TypeError: 'float' object is not iterable
    # print(jax.numpy.sum(1., 2., 3.))
    # TypeError: sum requires ndarray or scalar arguments, got <class 'list'> at position 0.
    # print(jax.numpy.sum([1., 2., 3.]))
    # TypeError: Cannot interpret value of type <class 'range'> as an abstract array; it does not have a dtype attribute
    # print(jax.numpy.sum(range(9)))
    print(jax.numpy.sum(jax.numpy.array([1., 2., 3.])))
    
def test():
    
    accumulate()
    
if __name__ == "__main__":
    
    test()