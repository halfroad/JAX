import jax


def print_log(x):

    print("print test:", x)

    x = jax.numpy.log(x)

    print("print test", x)

    return x

def main():

    jaxpr_print_log = jax.make_jaxpr(print_log)

    print(jaxpr_print_log(3.0))

if __name__ == '__main__':

    main()
