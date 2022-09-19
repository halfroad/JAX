import jax.numpy as jnp
from jax import lax


def compare():

    result = jnp.add(1, 1.0)

    print(result)

    result = lax.add(1, 1)

    print(result)

    # error
    # result = lax.add(1, 1.0)
    result = lax.add(jnp.float32(1), 1.0)

    print(result)


def main():

    compare()


if __name__ == "__main__":

    main()
