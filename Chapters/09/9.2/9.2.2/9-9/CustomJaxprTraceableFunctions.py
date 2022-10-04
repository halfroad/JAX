# Offer the backwards iterating solution
import jax
import jax.core

def inverse_jaxpr(jaxpr, consts, *args):

    inverse_registry = {jax.lax.exp_p: jax.numpy.log, jax.lax.tanh_p: jax.numpy.arctanh}

    environment = {}

    def read(var):

        if type(var) is jax.core.Literal:

            return var.val

        return environment[var]

    def write(var, val):

        environment[var] = val

    # Parameters are written onto Jaxpr outvars
    # Removed in newer version
    # write(jax.core.unitvar, jax.core.unit)

    jax.util.safe_map(write, jaxpr.outvars, args)
    jax.util.safe_map(write, jaxpr.constvars, consts)

    # Iterating backwards
    for equation in jaxpr.eqns[:: -1]:

        invals = jax.util.safe_map(read, equation.outvars)

        if equation.primitive not in inverse_registry:

            raise NotImplementedError("{} does not have registered inverse.".format(equation.primitive))

        outval = inverse_registry[equation.primitive](*invals)

        jax.util.safe_map(write, equation.invars, [outval])

    return jax.util.safe_map(read, jaxpr.invars)

# Build the backwards iteration in function
def inverse(function_):

    @jax.util.wraps(function_)
    def wrapped(*args, **kwargs):

        closed_jaxpr = jax.make_jaxpr(function_)(*args, **kwargs)
        out = inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)

        return out[0]

    return wrapped

def function(x):

    return jax.numpy.exp(jax.numpy.tanh(x))

print(jax.make_jaxpr(function)(1.))
print("---------------------------------")

inverse_function = inverse(function)

print(jax.make_jaxpr(inverse_function)(function(1.)))
