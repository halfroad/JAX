import jax


def loop_body(previous_i):

    return previous_i + 1

def g_inner_jitted(x, n):

    i = 0

    while i < n:

        # Don't use following function
        jit_loop_body = jax.jit(loop_body)
        i = jit_loop_body(i)

    return x + i

def main():

    result = g_inner_jitted(10, 20)

    print(result)

if __name__ == '__main__':

    main()
