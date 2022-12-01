import jax.example_libraries.optimizers

def init_random_params(key, intput_shape):

    initial_params = jax.random.normal(key = key, shape = intput_shape)

    return (key, intput_shape), initial_params

def loss_function(params, inputs, genuines):

    genuines_hat = predict(params, inputs)
    losses = genuines * jax.numpy.log(genuines_hat) + (1 - genuines) * jax.numpy.log(1 - genuines_hat)

    return -jax.numpy.mean(losses)

def sigmoid(inputs):

    return 0.5 * (jax.numpy.tanh(inputs / 2) + 1)

# Create the predictive model
def predict(params, inputs):

    outputs = jax.numpy.dot(inputs, params[0]) + params[1]

    return sigmoid(outputs)

def optimize(inputs, targets):

    optimizer_init, optmizer_update, get_params_function = jax.example_libraries.optimizers.adam(step_size = 2e-4)

    key = jax.random.PRNGKey(15)
    intput_shape = (28, 28)

    # Initialize the params
    _, initial_params = init_random_params(key, intput_shape)

    # Initialize the state
    optmizer_state = optimizer_init(initial_params)

    grad_loss_function = jax.grad(loss_function)
    params = get_params_function(optmizer_state)
    gradients = grad_loss_function(params, (inputs, targets))

    # Optimize the data and update the params using optimzier
    optmizer_state = optmizer_update(_, gradients, optmizer_state)
