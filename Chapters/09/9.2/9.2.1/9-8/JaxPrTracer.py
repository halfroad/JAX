import jax.random

@jax.jit
def function(x):

    return x + 1

def examine_jaxpr(closed_jaxpr):

    jaxpr = closed_jaxpr.jaxpr

    print("in_vars:", jaxpr.invars)
    print("out_vars:", jaxpr.outvars)
    print("const_vars:", jaxpr.constvars)

    for eqn in jaxpr.eqns:

        print("Equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)

    print("jaxpr:", jaxpr)

def start():

    x = jax.random.normal(jax.random.PRNGKey(0), (5000, 5000))

    jit_function = jax.jit(function)

    print(jit_function(x))

    jaxpr_function = jax.make_jaxpr(function)
    print(jaxpr_function(2.))

    print(examine_jaxpr(jaxpr_function(2.)))

    closed_jaxpr = jax.make_jaxpr(function)(2.)

    print(closed_jaxpr)
    print(closed_jaxpr.literals)


def main():

    start()

if __name__ == "__main__":

    main()
