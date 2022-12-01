import jax


def Dense(out_dimension, weight_init = jax.nn.initializers.glorot_normal(), bias_init = jax.random.normal(key = jax.random.PRNGKey(15), shape = (100,))):

    def init_funciton():

        return None

    def apply_function():

        return None

    return init_funciton, apply_function
