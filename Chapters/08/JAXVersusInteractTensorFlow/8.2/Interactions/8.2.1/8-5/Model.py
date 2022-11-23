import jax


def init_mlp_params(key, pixels, output_dimensions):

    params = []

    layer_dimensions = [pixels, 512, 256, output_dimensions]

    for i in range(1, len(layer_dimensions)):

        weight = jax.random.normal(key, shape = (layer_dimensions[i - 1], layer_dimensions[i])) / jax.numpy.sqrt(pixels)
        bias = jax.random.normal(key, shape = (layer_dimensions[i],)) / jax.numpy.sqrt(pixels)

        _dict = {"weight": weight, "bias": bias}

        params.append(_dict)

    return params

@jax.jit
def relu(inputs):

    return jax.numpy.maximum(0, inputs)

# Prediction
def forward(params, inputs):

    for param in params[: -1]:

        weight = param["weight"]
        bias = param["bias"]

        inputs = jax.numpy.dot(inputs, weight) + bias
        inputs = relu(inputs)

    outputs = jax.numpy.dot(inputs, params[-1]["weight"]) + params[-1]["bias"]

    print(outputs.shape)
    
    outputs = jax.nn.softmax(outputs, axis = -1)

    return outputs
