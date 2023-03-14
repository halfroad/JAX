import jax

def update_array_mimic_numpy():
    
    array = jax.numpy.zeros(shape = (3, 3))
    print("Original array:\n", array)
    
    array[1, :] = 1.
    
    print("Updated array:\n", array)
    
def update_array(array: jax.numpy.array):
    
    print("Original array = ", array)
    print("-----------------------------------")
    
    new_array = array.at[1, :].set(2.)
    print("new_array by at and set = ", new_array)
    print("-----------------------------------")
    
    new_array = array.at[1, :].add(5.)
    print("new_array by at and add = ", new_array)
    print("-----------------------------------")
    
    new_array = array.at[1, :].max(-1)
    print("new_array by at and max(-1) = ", new_array)
    
    print("-----------------------------------")
    new_array = array.at[1, :].max(1)
    print("new_array by at and max(1) = ", new_array)
    print("-----------------------------------")
    
    new_array = array.at[1, :].mul(10)
    print("new_array by at and mul(10) = ", new_array)
    print("-----------------------------------")
    
    
def test():
    
    # update_array_mimic_numpy()
    array = jax.numpy.ones(shape = (3, 3), dtype = jax.numpy.float32)
    update_array(array)
    
if __name__ == "__main__":
        
    test()