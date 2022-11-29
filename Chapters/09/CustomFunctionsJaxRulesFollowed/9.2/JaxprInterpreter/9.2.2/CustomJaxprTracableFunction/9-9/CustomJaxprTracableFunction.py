import jax


def inverse_jaxpr(jaxpr, consts, *args):

    environments = {}
    inverse_registry = {jax.lax.exp_p: jax.numpy.log, jax.lax.tanh_p: jax.numpy.arctanh}

    def read(variable):

        if type(variable) is jax.core.Literal:

            return variable.val

        return environments[variable]

    def write(variable, value):

        environments[variable] = value

    # Params written into Jaxpr outvars

    # write(jax.core.unitvar, jax.core.unit)

    jax.util.safe_map(write, jaxpr.outvars, args)
    jax.util.safe_map(write, jaxpr.constvars, consts)


    # Backwards iteration
    for eqn in jaxpr.eqns[:: - 1]:

        in_values = jax.util.safe_map(read, eqn.outvars)

        if eqn.primitive not in inverse_registry:

            raise NotImplementedError("{} does not have registered inverse.".format(eqn.primitive))

        out_values = inverse_registry[eqn.primitive](*in_values)
        jax.util.safe_map(write, eqn.invars, [out_values])

    return jax.util.safe_map(read, jaxpr.invars)

# Create the backwards inverse iteration
def inverse(function_, ):

    @jax.util.wraps(function_)

    def wrapped(*args, **kwargs):

        jaxpr = jax.make_jaxpr(function_)
        closed_jaxpr = jaxpr(*args, **kwargs)

        out = inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)

        return out[0]

    return wrapped

def function(x):

    tanh = jax.numpy.tanh(x)
    exp = jax.numpy.exp(tanh)

    return exp

def run():

    jaxpr = jax.make_jaxpr(function)
    closed_jaxpr = jaxpr(1.)

    print(closed_jaxpr)
    print("------------------------")

    inverse_function = inverse(function)

    jaxpr = jax.make_jaxpr(inverse_function)
    closed_jaxpr = jaxpr(function(1.))

    print(closed_jaxpr)

if __name__ == '__main__':

    run()
