import jax
import jax.numpy as jnp


def convolve(x, w):

    output = []

    print("x = {}, w = {}, len(x) = {}".format(x, w, len(x)))

    for i in range(1, len(x) - 1):

        # slices = array[start: end]
        slices = x[i - 1: i + 2]

        dotted = jnp.dot(slices, w)

        print("i = {}, slices = {}, dotted = {}".format(i, slices, dotted))

        output.append(dotted)

    return jnp.array(output)


def manually_batched_convolve(xs, ws):

    output = []

    print("xs = {}, ws = {}, xs.shape = {}, xs.shape[0] = {}".format(xs, ws, xs.shape, xs.shape[0]))

    for i in range(xs.shape[0]):

        convolved = convolve(xs[i], ws[i])

        print("xs[i] = {}, ws[i] = {}, convolved = {}".format(xs[i], ws[i], convolved))

        output.append(convolved)

    stacked = jnp.stack(output)

    print("output = {}, stacked = {}".format(output, stacked))

    return stacked


def manually_vectorized_convolve(xs, ws):

    output = []

    """
    xs = [[0 1 2 3 4]
    [0 1 2 3 4]],
    
    ws = [[2. 3. 4.]
    [2. 3. 4.]],
    
    xs.shape = (2, 5),
    xs.shape[0] = 2
    """
    # 1...4
    for i in range(1, xs.shape[-1] - 1):

        slices = xs[:, i - 1: i + 2]
        aggregation = jnp.sum(slices * ws, axis = 1)

        output.append(aggregation)

    stacked = jnp.stack(output, axis = 1)

    return stacked

def manually_vectorized_convolve1(xs, ws):

    output = []

    for i in range(1, xs.shape[-1] - 1):

        slices = xs[:, i - 1: i + 2] @ ws.T

        output.append(slices)

    return jnp.stack(output, axis = 1)

def start():

    x = jnp.arange(5)
    w = jnp.array([2., 3., 4.])

    result = convolve(x, w)

    print("convolve = ", result)
    print("-----------------------------------")

    """
    
    x = [0 1 2 3 4], w = [2. 3. 4.], len(x) = 5
    i = 1, slices = [0 1 2], dotted = 11.0
    i = 2, slices = [1 2 3], dotted = 20.0
    i = 3, slices = [2 3 4], dotted = 29.0
    convolve =  [11. 20. 29.]
    
    """

    xs = jnp.stack([x, x])
    ws = jnp.stack([w, w])

    result = manually_batched_convolve(xs, ws)

    print("manually_batched_convolve = ", result)

    result = manually_vectorized_convolve(xs, ws)

    print("manually_vectorized_convolve = ", result)

    result = manually_vectorized_convolve1(xs, ws)

    print("manually_vectorized_convolve1 = ", result)


def main():

    start()

if __name__ == "__main__":

    main()
