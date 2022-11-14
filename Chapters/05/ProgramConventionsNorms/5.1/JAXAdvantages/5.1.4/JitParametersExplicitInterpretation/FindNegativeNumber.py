from functools import partial

import jax


@jax.jit
def f(x, negative):

    return -x if negative else x

@partial(jax.jit, static_argnums = (1,))
def f2(x, negative):

    return -x if negative else x

def main():

    # Error
    # f(1, True)

    result = f2(1, True)

    print(f"Result = {result}")

if __name__ == "__main__":

    main()
