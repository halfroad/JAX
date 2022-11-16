"""

import jax


def function(x):

    if x > 0:
        return x
    else:
        return 2 * x

def main():

    jit_function = jax.jit(function)

    # jit_function(10)

    grad_function = jax.grad(function)

    print(grad_function(10.))

if __name__ == '__main__':

    main()


"""

import jax


def function(x):

    if x > 0:
        return x
    else:
        return 2 * x

def main():

    jit_function = jax.jit(function, static_argnums = (0,))

    print(jit_function(10.))

    jaxpr_function = jax.make_jaxpr(function, static_argnums = (0,))

    print(jaxpr_function(10.))

if __name__ == '__main__':

    main()
