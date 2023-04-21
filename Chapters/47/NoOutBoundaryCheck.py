import jax

def test():

    array = jax.numpy.arange(10)
    
    print("array =", array)
    print("array[15] = ", array[15])
    
def main():

    test()
    
if __name__ == "__main__":

    main()
