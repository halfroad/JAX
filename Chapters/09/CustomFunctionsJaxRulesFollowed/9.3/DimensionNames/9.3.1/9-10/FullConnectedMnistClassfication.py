import jax


def forward(weight1, weight2, images):

    dots = jax.numpy.dot(images, weight1)

    hidden_layer1 = jax.nn.relu(dots)
    hidden_layer2 = jax.numpy.dot(hidden_layer1, weight2)

    logtis = jax.nn.softmax(hidden_layer2)

    return logtis

def loss_function(weight1, weight2, images, labels):

    predictions = forward(weight1, weight2, images)
    targets = jax.nn.one_hot(labels, predictions.shape[-1])
    losses = jax.numpy.sum(targets * predictions, axis = 1)

    return -jax.numpy.mean(losses, axis = 0)

def train():

    weight1 = jax.numpy.zeros(shape = (784, 512))
    weight2 = jax.numpy.zeros(shape = (512, 10))

    images = jax.numpy.zeros(shape = (128, 784))
    labels = jax.numpy.zeros(shape = 128, dtype = jax.numpy.int32)

    losses = loss_function(weight1, weight2, images, labels)

    print(losses)

if __name__ == '__main__':

    train()
