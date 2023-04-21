import jax

def test():

    array = jax.numpy.arange(10)
    
    print("array =", array)
    print("array[5] = ", array[5])
    
    array_new = array.at[5].set(20)
    
    print("array =", array)
    print("array[5] = ", array[5])
    
    print("array_new =", array_new)
    print("array_new[5] = ", array_new[5])
    
def main():

    test()
    
if __name__ == "__main__":

    main()
