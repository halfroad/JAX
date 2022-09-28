import jax

# Single Image Pooling
def pooling(features, pool_size = 2, stride = 2):

    shape = features.shape

    height = shape[0]
    width = shape[1]

    padding_height = (round((height - pool_size + 1) / stride))
    padding_width = (round((width - pool_size + 1) / stride))

    pool_out = jax.numpy.zeros((padding_height, padding_width))

    out_height = 0

    for i in jax.numpy.arange(0, height, stride):

        out_width = 0

        for j in jax.numpy.arange(0, width, stride):

            pool_out = pool_out.at[out_height, out_width].set(jax.numpy.max(features[i: i + pool_size, j: j + pool_size]))

            out_height = out_height + 1
            out_width = out_width + 1

    return pool_out

def main():

    key = jax.random.PRNGKey(17)
    image = jax.random.normal(key, (10, 10))

    print(image)

    pool = pooling(image)

    print(pool)

if __name__ == "__main__":

    main()
