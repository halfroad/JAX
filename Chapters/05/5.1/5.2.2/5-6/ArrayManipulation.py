import jax
import jax.numpy as jnp

# Adjust the array
#from jax.ops import index, index_add, index_add, index_update, index_max, index_mul

def reshape_array():

    array = jnp.zeros((3, 3), dtype = jnp.float32)

    print(array)

    # Error
    # array[1, :] = 1.0

    print("Original array: ", array)
    print("---------------------")

    new_array = array.at[1, :].set(1.0)

    print("new_array: ", new_array)
    print("---------------------")

    new_add_array = array.at[1, :].add(1.0)

    print("new_add_array: ", new_add_array)
    print("---------------------")

    max_array = new_add_array.at[1, :].max(-1)

    print("negative_max_array: ", max_array)
    print("---------------------")

    max_array = new_add_array.at[1, :].max(1)

    print("positive_max_array: ", max_array)
    print("---------------------")

    mul_array = max_array.at[1, :].mul(2)

    print("mul_array: ", mul_array)
    print("---------------------")

def main():

    reshape_array()

if __name__ == "__main__":

    main()
