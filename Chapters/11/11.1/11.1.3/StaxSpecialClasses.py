import jax.numpy


def FanOut(number):

    """

    Layer contruction function for a fan-out layer.

    """

    def init_function():

        return lambda rng, input_shape: ([input_shape] * number, ())

    def apply_function():

        return lambda parameters, inputs, **kwargs: [inputs] * number

    return init_function, apply_function

def FanInsum():

    """

    Layer construction function for a fan-in sum layer.

    """

    def init_function():

        return lambda rng, input_shape: (input_shape[0], ())

    def apply_function():

        return lambda parameters, inputs, **kwrags: sum(inputs)

    return init_function, apply_function

def FanInConcat(axis = -1):

    """

    Layer construction function for a fan-in concatenation layer.

    """

    def init_function(rng, input_shape):

        ax = axis % len(input_shape[0])
        concat_size = sum(shape[ax] for shape in input_shape)
        out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax + 1:]

        return out_shape, ()

    def apply_function(parameters, inputs, **kwargs):

        return jax.numpy.concatenate(inputs, **kwargs)

    return init_function, apply_function

def Identify():

    """

    Layer construction function for an identity layer.

    """

    def init_function():

        return lambda rng, input_shape: (input_shape, ())

    def apply_function():

        return lambda parameters, inputs, **kwargs: inputs

    return init_function, apply_function


def start():

    FanOut(number = 2)
    FanInsum()
    FanInConcat()
    Identify()

def main():

    start()

if __name__ == "__main__":

    main()
