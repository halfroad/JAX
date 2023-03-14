import jax

def check_boundary():
    
    array = jax.numpy.arange(9)
    
    print(array)
    print(array[-1])
    print(array[11])
    
def test():
    
    check_boundary()
    
if __name__ == "__main__":
    
    test()