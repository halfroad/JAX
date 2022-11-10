import jax.numpy


def add():

    added = jax.numpy.add(1, 1.0)
    print(f"added = {added}")

    """
    
    added = jax.lax.add(1, 1)
    print(f"added = {added}")

    added = jax.lax.add(1, 1.0)
    print(f"added = {added}")
    
    """

    added = jax.lax.add(jax.numpy.float32(1), 1.0)
    print(f"added = {added}")

if __name__ == "__main__":

    add()
