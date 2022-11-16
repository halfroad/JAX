import jax


@jax.jit
def loop_body(previous_i):

    return previous_i + 1

def g_inner_jitted(x, n):

    i = 0

    while i < n:

        i = loop_body(i)

    return x + i

def main():

    result = g_inner_jitted(10, 20)

    print(result)

if __name__ == '__main__':

    main()
