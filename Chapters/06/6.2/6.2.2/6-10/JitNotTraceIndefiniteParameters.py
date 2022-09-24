import jax

def function(x):

    if x > 0:
        return x
    else:
        return 2 * x

def cond_function(input_):

    result = jax.lax.cond(input_ > 0, lambda x: x, lambda x: x + 1, input_)

    return result

@jax.jit
def loop_body(i):

    return i + 1

def g_inner_jitted(x, n):

    i = 0

    while i < n:

        i = loop_body(i)

    return x + 1

def start():

    """

    jit_function = jax.jit(function)
    result = jit_function(10.)

    print(result)

    """

    # grad

    grad_function = jax.grad(function)

    # No error, rule is comprimised by jit
    result = grad_function(10.)

    print("grad_function = ", result)

    # jit

    """

    jit_function = jax.jit(function)
    result = jit_function(10.)

    print(result)

    """

    jit_static_argnums_function = jax.jit(function, static_argnums = (0,))

    result = jit_static_argnums_function(10.)

    print("jit_static_argnums_function = ", result)

    jit_cond_function = jax.jit(cond_function)

    result = jit_cond_function(10.)

    print("jit_cond_function = ", result)

    # jaxpr

    jaxpr_jit_static_argnums_function = jax.make_jaxpr(jit_static_argnums_function)

    # Error
    # print(jaxpr_jit_static_argnums_function(10.))

    jaxpr_jit_cond_function = jax.make_jaxpr(jit_cond_function)

    print(jaxpr_jit_cond_function(10.))

    result = g_inner_jitted(10, 20)

    print("g_inner_jitted = ", result)

def main():

    start()

if __name__ == "__main__":

    main()
