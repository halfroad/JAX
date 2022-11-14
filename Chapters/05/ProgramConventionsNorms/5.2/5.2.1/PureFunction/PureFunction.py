import jax


def pure_uses_internel_state(x):

    state = dict(even = 0, odd = 0)

    for i in range(10):

        state["even" if i % 2 == 0 else "odd"] += x

    return state["even"] + state["odd"]

def main():

    jit_pure_uses_internel_state = jax.jit(pure_uses_internel_state)

    print(jit_pure_uses_internel_state(3.))

    jit_pure_uses_internel_state = jax.jit(pure_uses_internel_state)

    print(jit_pure_uses_internel_state(jax.numpy.array([5.])))


if __name__ == '__main__':

    main()
