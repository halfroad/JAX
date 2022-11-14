import jax

"""

Paragraph 5.1.1: run JAX as NumPy
Page 72

"""

def run_jax_like_numpy():

    array = jax.numpy.linspace(0, 9, 10)

    print(array)

def start():

    run_jax_like_numpy()


if __name__ == "__main__":

    start()
