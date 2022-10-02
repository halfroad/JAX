import jax.random
import sys

sys.path.append("../8-6/")
from JaxMnistModelComponents import relu

def init_parameters(number_pixel, output_dimensions):

    def init(layer_dimensions = [number_pixel, 512, 256, output_dimensions]):

        parameters = []

        key = jax.random.PRNGKey(17)

        for i in range(1, (len(layer_dimensions))):

            weight = jax.random.normal(key, shape = (layer_dimensions[i - 1], layer_dimensions[i])) / jax.numpy.sqrt(number_pixel)
            bias = jax.random.normal(key, shape = (layer_dimensions[i],)) / jax.numpy.sqrt(number_pixel)

            _dict = {"weight": weight, "bias": bias}

            parameters.append(_dict)

        return parameters

    return init()
