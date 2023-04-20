import jax
import numpy
from jax.experimental import maps

def predict(weight1, weight2, images):
    
    dots = jax.numpy.dot(images, weight1)
    hiddens = jax.nn.relu(dots)
    logtis = jax.numpy.dot(hiddens, weight2)
    
    return logtis - jax.nn.logsumexp(logtis, axis = 1, keepdims = True)

def loss_function(weight1, weight2, images, labels):
    
    predictions = predict(weight1 = weight1, weight2 = weight2, images = images)
    targets = jax.nn.one_hot(labels, predictions.shape[-1])
    losses = jax.numpy.sum(targets * predictions, axis = 1)
    
    return -jax.numpy.mean(losses, axis = 0)

# Named dimensions will be used to compute the data
def named_predict(weight1, weight2, images):
    
    pdot = jax.lax.pdot(images, weight1, "inputs")
    hidden = jax.nn.relu(pdot)
    logtis = jax.lax.pdot(hidden, weight2, "hidden")
    
    return logtis - jax.nn.logsumexp(logtis, "classes")

def named_loss_function(weight1, weight2, images, labels):
    
    predictions = named_predict(weight1, weight2, images)
    
    # jax.lax.psum(): Compute an all-reduce sum on x over the pmapped axis axis_name
    number_classes = jax.lax.psum(1, "classes")
    targets = jax.nn.one_hot(labels, number_classes, axis = "classes")
    losses = jax.lax.psum(targets * predictions, "classes")
    
    return -jax.lax.pmean(losses, "batch")
    
def train():
    
    weight1 = jax.numpy.zeros((784, 512))
    weight2 = jax.numpy.zeros((512, 10))
    
    images = jax.numpy.zeros((128, 784))
    labels = jax.numpy.zeros(128, dtype = jax.numpy.int32)
    
    losses = loss_function(weight1, weight2, images, labels)
    
    print("losses = ", losses)
    
    in_axes = [
        ["inputs", "hidden"],
        ["hidden", "classes"],
        ["batch", "inputs"],
        ["batch", ...]
        ]
    
    # Register the names for the dimensions
    loss_function_xmap = maps.xmap(named_loss_function, in_axes = in_axes, out_axes = [...], axis_resources = {"batch": "x"})
    
    devices = numpy.array(jax.local_devices())
    
    with jax.sharding.Mesh(devices, ("x",)):
        
        losses = loss_function_xmap(weight1, weight2, images, labels)
        
        print("losses = ", losses)
    
def main():
    
    train()
    
if __name__ == "__main__":
    
    main()