import jax


def function(x):

    return x + 1

def examine_jaxpr(closed_jaxpr):

    jaxpr = closed_jaxpr.jaxpr

    print("In Variables: ", jaxpr.invars)
    print("Out Varibales: ", jaxpr.outvars)
    print("Const Variables: ", jaxpr.constvars)

    print("-----------------------")

    for eqn in jaxpr.eqns:

        print("Equation: ", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)

    print("Jaxpr: ", jaxpr)

def run():

    jaxpr = jax.make_jaxpr(function)
    examine_jaxpr(jaxpr(2.))



if __name__ == '__main__':

    run()
