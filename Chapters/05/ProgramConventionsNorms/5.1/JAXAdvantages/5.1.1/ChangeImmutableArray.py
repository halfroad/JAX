import jax


def change_immutable_array():

    array = jax.numpy.linspace(0, 9, 10)

    new_array = array.at[0].set(17)

    print(f"array = {array}")
    print(f"new_array = {new_array}")

def start():

    change_immutable_array()

if __name__ == "__main__":

    start()
