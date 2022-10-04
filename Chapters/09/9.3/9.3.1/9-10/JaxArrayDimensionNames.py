import jax


def forward(weight1, weight2, images):

    hidden1 = jax.nn.relu(jax.numpy.dot(images, weight1))
    hidden2 = jax.numpy.dot(hidden1, weight2)

    logits = jax.nn.softmax(hidden2)

    return logits

def loss(weight1, weight2, images, labels):

    predictions = forward(weight1, weight2, images)
    targets = jax.nn.one_hot(labels, predictions.shape[-1])
    losses = jax.numpy.sum(targets * predictions, axis = 1)

    return -jax.numpy.mean(losses, axis = 0)

def named_predict(weight1, weight2, image):

    hidden1 = jax.nn.relu(jax.lax.pdot(image, weight1, "inputs"))
    logits = jax.lax.pdot(hidden1, weight2, "hidden1")

    return logits - jax.nn.logsumexp(logits, "classes")

def named_loss(weight1, weight2, images, labels):

    predictions = named_predict(weight1, weight2, images)
    number_classes = jax.lax.psum(1, "classes")
    targets = jax.nn.one_hot(labels, number_classes, axis = "classes")
    losses = jax.lax.psum(targets * predictions, "classes")

    return jax.lax.pmean(losses, "batch")

def start():
    
    weight1 = jax.numpy.zeros((784, 512))
    weight2 = jax.numpy.zeros((512, 10))

    images = jax.numpy.zeros((128, 784))
    labels = jax.numpy.zeros(128, dtype = jax.numpy.int32)

    losses = loss(weight1, weight2, images, labels)

    print(losses)

    axes = [
        ["inputs", "hidden1"],
        ["hidden1", "classes"],
        ["batch", "inputs"],
        ["batch", ...]
    ]

    # Register the dimension names
    loss_function = jax.experimental.maps.xmap(named_loss, in_axes = axes, out_axes = [...])
    losses = loss_function(weight1, weight2, images, labels)

    print(losses)

def main():

    start()

if __name__ == "__main__":

    main()
