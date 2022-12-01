def FanInSum():

    """
    Layer construction function for a fan-in sum layer.
    """

    init_function = lambda rng, input_shape: (input_shape[0], ())
    apply_funciton = lambda params, inputs, **kwargs: sum(inputs)

    return init_function, apply_funciton

def train():

    fanInSum = FanInSum()
