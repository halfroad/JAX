import jax


def function(x):

    return x + 1

def run():

    key = jax.random.PRNGKey(15)
    x = jax.random.normal(key, shape = (5000, 5000))

    fast_function = jax.jit(function)
    outputs = fast_function(x)

    print("outputs = ", outputs.shape)

if __name__ == '__main__':

    run()
