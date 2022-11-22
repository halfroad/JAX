import jax
import tensorflow_datasets as tfds

def setup():

    (trains, tests), info = tfds.load("mnist", split = [tfds.Split.TRAIN, tfds.Split.TEST], batch_size = -1, with_info = True, data_dir = "../../../../../../../Shares/Datasets/MNIST/")

    # Extract informative features
    class_names = info.features["label"].names
    n_classes = info.features["label"].num_classes

    trains = tfds.as_numpy(trains)
    tests = tfds.as_numpy(tests)

    train_images, train_labels = trains["image"], trains["label"]
    test_images, test_labels = tests["image"], tests["label"]

    return (n_classes, class_names), ((train_images, train_labels), (test_images, test_labels))

def one_hot_nojit(inputs, k = 10, dtype = jax.numpy.float32):

    matches = jax.numpy.array(inputs[:, None] == jax.numpy.arange(k), dtype)

    return matches

def batch_normalize(inputs, gamma = 0.9, beta = 0.25, epsilon = 1e-9):

    u = jax.numpy.mean(inputs, axis = 0)
    standard_variance = jax.numpy.sqrt(inputs.var(axis = 0) + epsilon)
    y = (inputs - u) / standard_variance
    scale_shift = y * gamma + beta

    return scale_shift

def conv(inputs, weights, window_strides = 1):

    shape = inputs.shape

    dimension_numbers = jax.lax.conv_dimension_numbers(shape, weights["kernel_weight"].shape, ("NHWC", "HWIO", "NHWC"))
    inputs = jax.lax.conv_general_dilated(inputs, weights["kernel_weight"], window_strides = [window_strides, window_strides], padding = "SAME", dimension_numbers = dimension_numbers)
    inputs = jax.nn.relu(inputs)

    return inputs

@jax.jit
def forward(params, inputs):

    for i in range(len(params) - 2):

        inputs = conv(inputs, weights = params[i])

    # Replace the pooling layer with batch_normalize layer
    inputs = batch_normalize(inputs)
    inputs = jax.numpy.reshape(inputs, [-1, inputs[0]])

    for i in range(len(params) - 2, len(params) - 1):

        inputs = jax.numpy.matmul(inputs, params[i]["weight"]) + params[i]["bias"]
        inputs = jax.nn.selu(inputs)

    inputs = jax.numpy.matmul(inputs, params[-1]["weight"]) + params[-1]["bias"]
    inputs = jax.nn.softmax(inputs, axis = -1)

    return inputs

def test():

    (n_classes, class_names), ((train_images, train_labels), (test_images, test_labels)) = setup()

    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

if __name__ == '__main__':

    test()
