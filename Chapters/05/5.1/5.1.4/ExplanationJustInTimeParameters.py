import jax
import jax.numpy as jnp

from functools import partial

def f(x, y):

    print("Running function f():")

    print(f"    x = {x}")
    print(f"    y = {y}")

    result = jnp.dot(x + 1, y + 1)

    print(f"    result = {result}")

    return result

@jax.jit
def func(x, negative):

    return -x if negative else x

@partial(jax.jit, static_argnums= (1,))
def func2(x, negative):

    return -x if negative else x

g = 0

def impure_saves_global(x):

    global g
    g = x

    return x

def pure_uses_internal_state(x):

    state = dict(even = 0, odd = 0)

    for i in range(10):

        state["even" if i % 2 == 0 else "odd"] += x

    return state["even"] + state["odd"]

def main():

    key = jax.random.PRNGKey(1)

    x = jax.random.normal(key, shape = [5, 3])
    y = jax.random.normal(key, shape = [3, 4])

    f(x, y)

    print("--------------------")

    jax.jit(f)(x, y)

    a = jnp.array([[1, 2, 4],
         [2, 3, -1],
         [-1, -2, 0],
         [3, 4, -2],
         [3, 1, 4]])

    b = jnp.array([[1, 5, 2, 4],
         [3, 2, 4, 1],
         [4, -1, 3, 0]])

    print(jnp.dot(a, b))

    # Error
    # func(1, True)
    print(func2(1, True))

    print("First call:", jax.jit(impure_saves_global)(4.0))
    print("Saved global: ", g)

    print(jax.jit(pure_uses_internal_state)(3.0))
    print(jax.jit(pure_uses_internal_state)(jnp.array([5.0])))

if __name__ == "__main__":

    main()
