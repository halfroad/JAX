import numpy as np
import jax.numpy as jnp

def in_place_modify(inputs):

    inputs[0] = 123

    return None

def start():

    x = np.array([1, 2, 3])

    in_place_modify(x)

    print(x)

    """
    Error
    
    x = jnp.array([1, 2, 3])

    in_place_modify(x)

    print(x)
    
    """

def main():

    start()

if __name__ == "__main__":

    start()
