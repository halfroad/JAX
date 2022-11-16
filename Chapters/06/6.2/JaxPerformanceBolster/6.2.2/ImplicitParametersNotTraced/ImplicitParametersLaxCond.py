import jax.lax


def function(x):

    result = jax.lax.cond(x > 0, lambda x: x, lambda x: x + 1, x)

    return result

def main():

    jit_function = jax.jit(function)
    result = jit_function(10.0)

    print(result)

    jaxpr_function = jax.make_jaxpr(function)

    print(jaxpr_function(10.))

if __name__ == '__main__':

    main()
