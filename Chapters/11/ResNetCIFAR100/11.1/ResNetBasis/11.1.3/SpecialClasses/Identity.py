def Identity():

    """
    Layer construction function for an identity layer.
    """
    init_function = lambda rng, input_shape: (input_shape, ())
    apply_function = lambda params, inputs, **kwrags: inputs

    return init_function, apply_function

def train():

    identity = Identity()
