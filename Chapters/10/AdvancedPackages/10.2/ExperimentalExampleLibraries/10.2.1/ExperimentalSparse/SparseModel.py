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

    return (key, num_classes, classes, params), (inputs, sparsed_inputs, genuines)

def train():

    (key, num_classes, classes, params), (inputs, sparsed_inputs, genuines) = setup()

    print((key, num_classes, classes, params), (inputs, sparsed_inputs, genuines))

if __name__ == '__main__':

    train()
