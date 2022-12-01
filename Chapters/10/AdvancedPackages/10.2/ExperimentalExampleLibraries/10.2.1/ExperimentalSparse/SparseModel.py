import jax
from jax.experimental import sparse


def setup():

    key = jax.random.PRNGKey(15)

    # 10 classifications
    num_classes = 10

    # Generate the series of classifcations
    classes = jax.numpy.arange(num_classes)

    inputs = []
    genuines = []

    for i in range(1024):

        input_ = jax.random.choice(key = (key + 1), a = classes, shape = (1,))[0]
        input_one_hot = jax.nn.one_hot(input_, num_classes = num_classes)

        inputs.append(input_one_hot)
        genuines.append(input_)

    params = [jax.random.normal(key, shape = (num_classes, 1)), jax.random.normal(key, shape = (1,))]
    # Convert to sparse array
    sparsed_inputs = sparse.BCOO.fromdense(jax.numpy.array(inputs))
    genuines = jax.numpy.array(genuines)

    learning_rate = 1e-3

    return (key, num_classes, classes, params, learning_rate), (inputs, sparsed_inputs, genuines)

# Create Sigmiod function
def sigmoid(inputs):

    return 0.5 * (jax.numpy.tanh(inputs / 2) + 1)

# Create the predictive model
def predict(params, inputs):

    outputs = jax.numpy.dot(inputs, params[0]) + params[1]

    return sigmoid(outputs)

# Loss Function
def loss_function(params, sparsed_inputs, genuines):

    sparsed_predict = sparse.sparsify(predict)
    genuines_hat = sparsed_predict(params, sparsed_inputs)
    losses = genuines * jax.numpy.log(genuines_hat) + (1 - genuines) * jax.numpy.log(1 - genuines_hat)

    return -jax.numpy.mean(losses)

def train():

    (key, num_classes, classes, params, learning_rate), (inputs, sparsed_inputs, genuines) = setup()

    loss = loss_function(params, sparsed_inputs, genuines)

    print("Loss = ", loss)

    for i in range(100):

        grad_loss_function = jax.grad(loss_function)
        gradients = grad_loss_function(params, sparsed_inputs, genuines)
        params = [(param - gradient * learning_rate) for param, gradient in zip(params, gradients)]

        print(f"Epoch {i}")


    loss = loss_function(params, sparsed_inputs, genuines)

    print("Loss = ", loss)


if __name__ == '__main__':

    train()
