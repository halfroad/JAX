import jax


def function(x):

    return x + 1

def trace_function():

    jaxpr_function = jax.make_jaxpr(function)
    close_jaxpr = jaxpr_function(1.0)

    print(close_jaxpr)
    print(close_jaxpr.literals)

def run():

    trace_function()

if __name__ == '__main__':

    run()
