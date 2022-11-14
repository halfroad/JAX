import jax

def update_array():

    array = jax.numpy.zeros((3, 3), dtype = jax.numpy.float32)

    print("Orginal array = ", array)
    print("-------------------------------")

    # Set the element to 1. at row #1
    new_array = array.at[1, :].set(1.)

    print("new array = ", new_array)
    print("-------------------------------")

    # Plus 1. to the element at row #1
    new_index_add_array = array.at[1, :].add(1.)

    print("new_index_add_array = ", new_index_add_array)
    print("-------------------------------")

    new_index_max_array = array.at[1, :].max(-1)

    print("new_index_max_array = ", new_index_max_array)
    print("-------------------------------")

    new_index_max_array = array.at[1, :].max(1)

    print("new_index_max_array = ", new_index_max_array)
    print("-------------------------------")

    mul_array = array.at[1, :].mul(2)

    print("mul_array = ", mul_array)
    print("-------------------------------")

if __name__ == '__main__':

    update_array()
