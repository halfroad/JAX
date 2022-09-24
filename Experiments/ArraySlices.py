import jax.numpy as jnp

def main():

    array = jnp.arange(10)

    print(array)
    print("-----------------------------------")

    print(array[:])
    print("-----------------------------------")

    print(array[0:])
    print("-----------------------------------")

    print(array[-1:])
    print("-----------------------------------")

    print(array[-3:])
    print("-----------------------------------")

    print(array[0: 4])
    print("-----------------------------------")

    print(array[0: -1])
    print("-----------------------------------")

    print(array[0: -2])
    print("-----------------------------------")

    array2 = jnp.arange(10, 20)
    array3 = jnp.arange(20, 30)

    array = [array, array2, array3]
    array = jnp.stack(array)

    print(array)
    print("-----------------------------------")

    print(array[:, 1: 2])
    print("-----------------------------------")

    print(array[1:, 1: 3])
    print("-----------------------------------")

    print(array[1:2, 1: 3])
    print("-----------------------------------")

if __name__ == "__main__":

    main()
