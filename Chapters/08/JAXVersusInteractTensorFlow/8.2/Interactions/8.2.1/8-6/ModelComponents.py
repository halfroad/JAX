import jax.numpy

@jax.jit
def relu(inputs):

    return jax.numpy.maximum(0, inputs)

@jax.jit
def cross_entropy(genuines, predictions):

    entropys = genuines * jax.numpy.log(jax.numpy.clip(predictions, 1e-9, 0.999)) + (1 - genuines) * jax.numpy.log(jax.numpy.clip(1 - predictions, 1e-9, 0.999))
    entropys = -jax.numpy.sum(entropys)

    return jax.numpy.mean(entropys)

@jax.jit
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

@jax.jit
def loss_function(params, inputs, genuines):

    predictions = forward(params, inputs)

    return cross_entropy(genuines, predictions)

@jax.jit
def optimizer(params, inputs, genuines, learning_rate = 1e-3):

    grad_loss_function = jax.grad(loss_function)
    gradients = grad_loss_function(params, inputs, genuines)

    params = jax.tree_util.tree_map(lambda param, gradient: param - learning_rate * gradient, params, gradients)

    return params

@jax.jit
def verify_accuracy(params, inputs, targets):

    result = forward(params, inputs)
    class_ = jax.numpy.argmax(result, axis = 1)
    targets = jax.numpy.argmax(targets, axis = 1)

    equals = jax.numpy.sum(class_ == targets)

    return equals
