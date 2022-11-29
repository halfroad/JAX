import jax


def function(x):

    return x + 1

def run():

    key = jax.random.PRNGKey(15)
    x = jax.random.normal(key, shape = (5000, 5000))

    fast_function = jax.jit(function)
    outputs = fast_function(x)

    print("outputs = ", outputs.shape)

    jaxpr = jax.make_jaxpr(function)
    print("jaxpr(2.0) = ", jaxpr(2.0))

if __name__ == '__main__':

    run()
