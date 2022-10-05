import jax.random
import jax.experimental.sparse


def setup():

    key = jax.random.PRNGKey(17)
    number_classes = 10
    classes = jax.numpy.arange(number_classes)

    inputs = []
    genuines = []

    for i in range(1024):

        # Create item randomly
        item = jax.random.choice(key + 1, classes, shape = (1, ))[0]

        # One-hoted the item
        item_one_hot = jax.nn.one_hot(item, num_classes = number_classes)

        inputs.append(item_one_hot)
        genuines.append(item)

    # Generate the parameters for model
    parameters = [jax.random.normal(key, shape = (number_classes, 1)), jax.random.normal(key, shape = (1, ))]

    # Sparsify the inputs into sparse matrix
    sparsed_inputs = jax.experimental.sparse.BCOO.fromdense(jax.numpy.array(inputs))
    genuines = jax.numpy.array(genuines)

    return sparsed_inputs, genuines, parameters, number_classes, classes

def start():

    sparsed_inputs, genuines, parameters, number_classes, classes = setup()

    print(sparsed_inputs, genuines, parameters, number_classes, classes)

def main():

    start()

if __name__ == "__main__":

    main()
