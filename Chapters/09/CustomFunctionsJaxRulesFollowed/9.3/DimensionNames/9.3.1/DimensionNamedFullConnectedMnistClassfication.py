import jax


def setup():

    weight1 = jax.numpy.zeros(shape = (784, 512))               # weight1 params dimension
    weight2 = jax.numpy.zeros(shape = (512, 10))                # weight1 params dimension

    images = jax.numpy.zeros((128, 784))                        # Inputs dimension
    labels = jax.numpy.zeros(128, dtype = jax.numpy.int32)      # Labels dimension

    in_axes = [

        ["inputs", "hidden"],
        ["hidden", "classes"],
        ["batch", "inputs"],
        ["batch", ...]
    ]

    return (weight1, weight2), (images, labels), in_axes

# Use dimension names to compute
def named_predict(weight1, weight2, images):

    pdot = jax.lax.pdot(images, weight1, "inputs")
    hidden = jax.nn.relu(pdot)
    logits = jax.lax.pdot(hidden, weight2, "hidden")

    return logits - jax.nn.logsumexp(logits, "classes")

def named_loss_function(weight1, weight2, images, labels):

    predictions = named_predict(weight1, weight2, images)
    num_classes = jax.lax.psum(1, "classes")
    targets = jax.nn.one_hot(labels, num_classes, axis = "classes")
    losses = jax.lax.psum(targets * predictions, "classes")

    return - jax.lax.pmean(losses, "batch")

def train():

    (weight1, weight2), (images, labels), in_axes = setup()

    # Register the named dimensions using xmap function
    map_named_loss_function = jax.experimental.maps.xmap(named_loss_function, in_axes = in_axes, out_axes = [...])
    losses = map_named_loss_function(weight1, weight2, images, labels)
    print("losses = ", losses)

if __name__ == '__main__':

    train()
