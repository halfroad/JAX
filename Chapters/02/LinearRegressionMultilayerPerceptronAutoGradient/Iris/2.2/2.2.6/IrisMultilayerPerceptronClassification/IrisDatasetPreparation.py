import jax
import sklearn.datasets

"""

Paragraph 2.2.6, Experiment on Iris Classification based Multilayer Perceptron
Page 32
First Step: Dataset Preparation

"""

def setup():

    iris = sklearn.datasets.load_iris()

    data = jax.numpy.float32(iris.data)
    targets = jax.numpy.float32(iris.target)

    prng = jax.random.PRNGKey(15)

    data = jax.random.permutation(prng, data, independent = True)
    targets = jax.random.permutation(prng, targets, independent = True)

    targets = one_hot_nojit(targets)

    prng = jax.random.PRNGKey(15)
    epochs = 20000

    w0 = jax.random.normal(prng, shape = [4, 5])
    b0 = jax.random.normal(prng, shape = (5,))

    w1 = jax.random.normal(prng, shape = [5, 10])
    b1 = jax.random.normal(prng, shape = (10,))

    parameters = [w0, b0, w1, b1]
    learning_rate = 2.17e-4

    return data, targets, prng, epochs, parameters, learning_rate

def one_hot_nojit(inputs, size = 10, dtype = jax.numpy.float32):

    """

    Create an one-hot encoding for inputs with the size.

    """
    slices = inputs[:, None]
    array = jax.numpy.arange(size)

    return jax.numpy.array(slices == array, dtype = dtype)

def start():

    data, targets = setup()

    print(f"Data = {data[: 10]}\nTargets = {targets[: 10]}")

if __name__ == "__main__":

    start()
