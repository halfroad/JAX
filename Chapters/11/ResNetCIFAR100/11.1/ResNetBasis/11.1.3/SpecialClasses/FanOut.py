def FanOut(number):

    """
    Layer construction function for a fan-out layer.
    """

    init_function = lambda rng, input_shape: ([input_shape] * number, ())
    apply_function = lambda params, inputs, **kwargs: [inputs] * number

    return init_function, apply_function

def train():

    init_function, apply_function = FanOut(number = 2)
