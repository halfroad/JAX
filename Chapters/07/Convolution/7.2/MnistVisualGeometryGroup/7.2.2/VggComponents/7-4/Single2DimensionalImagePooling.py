import jax


def pooling(features, pool_size = 2, stride = 2):

    """

    Be noted that this fucntion only can be used on the pooling for single image,
    cannot be used on batch circumstances

    """

    features_shape = features.shape

    height = features_shape[0]
    width = features_shape[1]

    padding_height = round((height - pool_size + 1) / stride)
    padding_width = round((width - pool_size + 1) / stride)

    pool_output = jax.numpy.zeros(shape = (padding_height, padding_width))
    out_height = 0

    for i in jax.numpy.arange(0, height, stride):

        out_width = 0

        for j in jax.numpy.arange(0, width, stride):

            pool_output = pool_output.at[out_height, out_width].set(jax.numpy.max(features[i: i + pool_size, j: j + pool_size]))

            out_width = out_width + 1
            out_height = out_height + 1

    return pool_output

def test():

    key = jax.random.PRNGKey(15)

    features = jax.random.normal(key, shape = (200, 200))

    pool_output = pooling(features)

    print(pool_output)

if __name__ == '__main__':

    test()
