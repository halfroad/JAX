import jax

def BatchNormalization(axis = (0, 1, 2), epsilon = 1e-5, center = True, scale = True, beta_init = jax.numpy.zeros(shape = (100,)), gamma_init = jax.numpy.ones(shape = (100,))):

    def init_function():

        return None

    def apply_function():

        return None

    return init_function, apply_function

