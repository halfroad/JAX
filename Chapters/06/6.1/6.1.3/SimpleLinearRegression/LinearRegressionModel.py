import jax


def model(params, inputs):

    """

    Create the model

    """

    weight = params[0]
    bias = params[1]

    predictions = inputs * weight + bias

    return predictions

def loss_fucntion(params, inputs, genuines):

    prediction = model(params, inputs)
    loss = jax.numpy.mean((prediction - genuines) ** 2)

    return loss

def update(params, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_fucntion)
    gradient = grad_loss_function(params, inputs, genuines)

    params = params - learning_rate * gradient

    return params
