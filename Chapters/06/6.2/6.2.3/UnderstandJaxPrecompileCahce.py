import jax

def loop_body(i):

    return i + 1

def g_inner_jitted(x, n):

    i = 0

    while x < n:

        i = jax.jit(loop_body)(i)

        return x + 1

def main():

    result = g_inner_jitted(10, 20)

    print("g_inner_jitted = ", result)

if __name__ == "__main__":

    main()


